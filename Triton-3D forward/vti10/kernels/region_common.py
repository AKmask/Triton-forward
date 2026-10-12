"""Shared staggered derivatives and interior updates for 3-D VTI kernels.

Flattened x/y/z strides are 1, nx, and nx*ny. Region indices exclude
the stencil halo; inactive lanes are masked during memory operations.
"""

from __future__ import annotations
import triton
import triton.language as tl


@triton.jit
def _fwd(p, idx, active, stride: tl.constexpr, invh,
         c0: tl.constexpr, c1: tl.constexpr, c2: tl.constexpr,
         c3: tl.constexpr, c4: tl.constexpr, HALF: tl.constexpr):
    # Evaluate the derivative at the forward half-grid position.
    r = c0*(tl.load(p+idx+stride,mask=active,other=0.0)-tl.load(p+idx,mask=active,other=0.0))
    if HALF >= 2:
        r += c1*(tl.load(p+idx+2*stride,mask=active,other=0.0)-tl.load(p+idx-stride,mask=active,other=0.0))
    if HALF >= 3:
        r += c2*(tl.load(p+idx+3*stride,mask=active,other=0.0)-tl.load(p+idx-2*stride,mask=active,other=0.0))
    if HALF >= 4:
        r += c3*(tl.load(p+idx+4*stride,mask=active,other=0.0)-tl.load(p+idx-3*stride,mask=active,other=0.0))
    if HALF >= 5:
        r += c4*(tl.load(p+idx+5*stride,mask=active,other=0.0)-tl.load(p+idx-4*stride,mask=active,other=0.0))
    return r*invh


@triton.jit
def _bwd(p, idx, active, stride: tl.constexpr, invh,
         c0: tl.constexpr, c1: tl.constexpr, c2: tl.constexpr,
         c3: tl.constexpr, c4: tl.constexpr, HALF: tl.constexpr):
    # Evaluate the derivative at the backward half-grid position.
    r = c0*(tl.load(p+idx,mask=active,other=0.0)-tl.load(p+idx-stride,mask=active,other=0.0))
    if HALF >= 2:
        r += c1*(tl.load(p+idx+stride,mask=active,other=0.0)-tl.load(p+idx-2*stride,mask=active,other=0.0))
    if HALF >= 3:
        r += c2*(tl.load(p+idx+2*stride,mask=active,other=0.0)-tl.load(p+idx-3*stride,mask=active,other=0.0))
    if HALF >= 4:
        r += c3*(tl.load(p+idx+3*stride,mask=active,other=0.0)-tl.load(p+idx-4*stride,mask=active,other=0.0))
    if HALF >= 5:
        r += c4*(tl.load(p+idx+4*stride,mask=active,other=0.0)-tl.load(p+idx-5*stride,mask=active,other=0.0))
    return r*invh


@triton.jit
def interior_velocity_kernel(indices, npoints,
                             vx,vy,vz,sxx,syy,szz,sxy,sxz,syz,bx,by,bz,
                             nx:tl.constexpr,ny:tl.constexpr,nz:tl.constexpr,
                             dx,dy,dz,dt,
                             c0:tl.constexpr,c1:tl.constexpr,c2:tl.constexpr,c3:tl.constexpr,c4:tl.constexpr,
                             HALF:tl.constexpr,BLOCK:tl.constexpr):
    # Use stress derivatives to update velocities only at interior region indices.
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); active=i<npoints
    idx=tl.load(indices+i,mask=active,other=0)
    plane=nx*ny
    q0=_fwd(sxx,idx,active,1,1.0/dx,c0,c1,c2,c3,c4,HALF)
    q1=_bwd(sxy,idx,active,nx,1.0/dy,c0,c1,c2,c3,c4,HALF)
    q2=_bwd(sxz,idx,active,plane,1.0/dz,c0,c1,c2,c3,c4,HALF)
    q3=_bwd(sxy,idx,active,1,1.0/dx,c0,c1,c2,c3,c4,HALF)
    q4=_fwd(syy,idx,active,nx,1.0/dy,c0,c1,c2,c3,c4,HALF)
    q5=_bwd(syz,idx,active,plane,1.0/dz,c0,c1,c2,c3,c4,HALF)
    q6=_bwd(sxz,idx,active,1,1.0/dx,c0,c1,c2,c3,c4,HALF)
    q7=_bwd(syz,idx,active,nx,1.0/dy,c0,c1,c2,c3,c4,HALF)
    q8=_fwd(szz,idx,active,plane,1.0/dz,c0,c1,c2,c3,c4,HALF)
    Bx=tl.load(bx+idx,mask=active,other=0.0); By=tl.load(by+idx,mask=active,other=0.0); Bz=tl.load(bz+idx,mask=active,other=0.0)
    tl.store(vx+idx,tl.load(vx+idx,mask=active,other=0.0)+dt*Bx*(q0+q1+q2),mask=active)
    tl.store(vy+idx,tl.load(vy+idx,mask=active,other=0.0)+dt*By*(q3+q4+q5),mask=active)
    tl.store(vz+idx,tl.load(vz+idx,mask=active,other=0.0)+dt*Bz*(q6+q7+q8),mask=active)


@triton.jit
def interior_stress_kernel(indices,npoints,
                           vx,vy,vz,sxx,syy,szz,sxy,sxz,syz,
                           c11,c12,c13,c22,c23,c33,c44,c55,c66,
                           nx:tl.constexpr,ny:tl.constexpr,nz:tl.constexpr,
                           dx,dy,dz,dt,
                           c0:tl.constexpr,c1:tl.constexpr,c2:tl.constexpr,c3:tl.constexpr,c4:tl.constexpr,
                           HALF:tl.constexpr,BLOCK:tl.constexpr):
    # Apply the VTI stiffness tensor to staggered velocity derivatives.
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); active=i<npoints
    idx=tl.load(indices+i,mask=active,other=0); plane=nx*ny
    dxvx=_bwd(vx,idx,active,1,1.0/dx,c0,c1,c2,c3,c4,HALF)
    dyvy=_bwd(vy,idx,active,nx,1.0/dy,c0,c1,c2,c3,c4,HALF)
    dzvz=_bwd(vz,idx,active,plane,1.0/dz,c0,c1,c2,c3,c4,HALF)
    dxfvy=_fwd(vy,idx,active,1,1.0/dx,c0,c1,c2,c3,c4,HALF)
    dyfvx=_fwd(vx,idx,active,nx,1.0/dy,c0,c1,c2,c3,c4,HALF)
    dxfvz=_fwd(vz,idx,active,1,1.0/dx,c0,c1,c2,c3,c4,HALF)
    dzfvx=_fwd(vx,idx,active,plane,1.0/dz,c0,c1,c2,c3,c4,HALF)
    dyfvz=_fwd(vz,idx,active,nx,1.0/dy,c0,c1,c2,c3,c4,HALF)
    dzfvy=_fwd(vy,idx,active,plane,1.0/dz,c0,c1,c2,c3,c4,HALF)
    C11=tl.load(c11+idx,mask=active,other=0.0); C12=tl.load(c12+idx,mask=active,other=0.0); C13=tl.load(c13+idx,mask=active,other=0.0)
    C22=tl.load(c22+idx,mask=active,other=0.0); C23=tl.load(c23+idx,mask=active,other=0.0); C33=tl.load(c33+idx,mask=active,other=0.0)
    C44=tl.load(c44+idx,mask=active,other=0.0); C55=tl.load(c55+idx,mask=active,other=0.0); C66=tl.load(c66+idx,mask=active,other=0.0)
    tl.store(sxx+idx,tl.load(sxx+idx,mask=active,other=0.0)+dt*(C11*dxvx+C12*dyvy+C13*dzvz),mask=active)
    tl.store(syy+idx,tl.load(syy+idx,mask=active,other=0.0)+dt*(C12*dxvx+C22*dyvy+C23*dzvz),mask=active)
    tl.store(szz+idx,tl.load(szz+idx,mask=active,other=0.0)+dt*(C13*dxvx+C23*dyvy+C33*dzvz),mask=active)
    tl.store(sxy+idx,tl.load(sxy+idx,mask=active,other=0.0)+dt*C66*(dxfvy+dyfvx),mask=active)
    tl.store(sxz+idx,tl.load(sxz+idx,mask=active,other=0.0)+dt*C55*(dxfvz+dzfvx),mask=active)
    tl.store(syz+idx,tl.load(syz+idx,mask=active,other=0.0)+dt*C44*(dyfvz+dzfvy),mask=active)


def launch_interior_velocity(indices,fields,params,shape,spacing,dt,coeffs,half,block=256):
    n=int(indices.numel())
    if n==0: return
    nz,ny,nx=shape; dx,dy,dz=spacing
    interior_velocity_kernel[(triton.cdiv(n,block),)](indices,n,*fields,*params,nx,ny,nz,dx,dy,dz,dt,*coeffs,HALF=half,BLOCK=block)


def launch_interior_stress(indices,fields,cij,shape,spacing,dt,coeffs,half,block=256):
    n=int(indices.numel())
    if n==0: return
    nz,ny,nx=shape; dx,dy,dz=spacing
    interior_stress_kernel[(triton.cdiv(n,block),)](indices,n,*fields,*cij,nx,ny,nz,dx,dy,dz,dt,*coeffs,HALF=half,BLOCK=block)
