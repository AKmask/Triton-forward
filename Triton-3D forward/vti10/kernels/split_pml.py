"""Fuse derivatives, damping recurrences, and 3-D PML field reconstruction.

Nine velocity and fifteen stress split states retain the contribution
of each derivative direction. All fields use flattened [z, y, x] order.
"""

from __future__ import annotations
import triton
import triton.language as tl
from .region_common import _fwd, _bwd


@triton.jit
def pml_velocity_fused_kernel(indices,npoints,
                              vx,vy,vz,sxx,syy,szz,sxy,sxz,syz,bx,by,bz,
                              sv0,sv1,sv2,sv3,sv4,sv5,sv6,sv7,sv8,
                              pmlx,pmly,pmlz,
                              nx:tl.constexpr,ny:tl.constexpr,nz:tl.constexpr,
                              dx,dy,dz,dt,
                              c0:tl.constexpr,c1:tl.constexpr,c2:tl.constexpr,c3:tl.constexpr,c4:tl.constexpr,
                              HALF:tl.constexpr,BLOCK:tl.constexpr):
    # Each velocity has x/y/z split states; sum all three after damping.
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); active=i<npoints
    idx=tl.load(indices+i,mask=active,other=0); plane=nx*ny
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
    px=tl.load(pmlx+idx,mask=active,other=0.0); py=tl.load(pmly+idx,mask=active,other=0.0); pz=tl.load(pmlz+idx,mask=active,other=0.0)
    axn=1.0-0.5*dt*px; axd=1.0+0.5*dt*px; ayn=1.0-0.5*dt*py; ayd=1.0+0.5*dt*py; azn=1.0-0.5*dt*pz; azd=1.0+0.5*dt*pz
    a0=(axn*tl.load(sv0+idx,mask=active,other=0.0)+dt*Bx*q0)/axd
    a1=(ayn*tl.load(sv1+idx,mask=active,other=0.0)+dt*Bx*q1)/ayd
    a2=(azn*tl.load(sv2+idx,mask=active,other=0.0)+dt*Bx*q2)/azd
    a3=(axn*tl.load(sv3+idx,mask=active,other=0.0)+dt*By*q3)/axd
    a4=(ayn*tl.load(sv4+idx,mask=active,other=0.0)+dt*By*q4)/ayd
    a5=(azn*tl.load(sv5+idx,mask=active,other=0.0)+dt*By*q5)/azd
    a6=(axn*tl.load(sv6+idx,mask=active,other=0.0)+dt*Bz*q6)/axd
    a7=(ayn*tl.load(sv7+idx,mask=active,other=0.0)+dt*Bz*q7)/ayd
    a8=(azn*tl.load(sv8+idx,mask=active,other=0.0)+dt*Bz*q8)/azd
    tl.store(sv0+idx,a0,mask=active); tl.store(sv1+idx,a1,mask=active); tl.store(sv2+idx,a2,mask=active)
    tl.store(sv3+idx,a3,mask=active); tl.store(sv4+idx,a4,mask=active); tl.store(sv5+idx,a5,mask=active)
    tl.store(sv6+idx,a6,mask=active); tl.store(sv7+idx,a7,mask=active); tl.store(sv8+idx,a8,mask=active)
    tl.store(vx+idx,a0+a1+a2,mask=active); tl.store(vy+idx,a3+a4+a5,mask=active); tl.store(vz+idx,a6+a7+a8,mask=active)


@triton.jit
def pml_stress_fused_kernel(indices,npoints,
                            vx,vy,vz,sxx,syy,szz,sxy,sxz,syz,
                            c11,c12,c13,c22,c23,c33,c44,c55,c66,
                            ss0,ss1,ss2,ss3,ss4,ss5,ss6,ss7,ss8,ss9,ss10,ss11,ss12,ss13,ss14,
                            pmlx,pmly,pmlz,
                            nx:tl.constexpr,ny:tl.constexpr,nz:tl.constexpr,
                            dx,dy,dz,dt,
                            c0:tl.constexpr,c1:tl.constexpr,c2:tl.constexpr,c3:tl.constexpr,c4:tl.constexpr,
                            HALF:tl.constexpr,BLOCK:tl.constexpr):
    # Normal stresses use three directional states; shear stresses use two.
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
    px=tl.load(pmlx+idx,mask=active,other=0.0); py=tl.load(pmly+idx,mask=active,other=0.0); pz=tl.load(pmlz+idx,mask=active,other=0.0)
    axn=1.0-0.5*dt*px; axd=1.0+0.5*dt*px; ayn=1.0-0.5*dt*py; ayd=1.0+0.5*dt*py; azn=1.0-0.5*dt*pz; azd=1.0+0.5*dt*pz
    a0=(axn*tl.load(ss0+idx,mask=active,other=0.0)+dt*C11*dxvx)/axd
    a1=(ayn*tl.load(ss1+idx,mask=active,other=0.0)+dt*C12*dyvy)/ayd
    a2=(azn*tl.load(ss2+idx,mask=active,other=0.0)+dt*C13*dzvz)/azd
    a3=(axn*tl.load(ss3+idx,mask=active,other=0.0)+dt*C12*dxvx)/axd
    a4=(ayn*tl.load(ss4+idx,mask=active,other=0.0)+dt*C22*dyvy)/ayd
    a5=(azn*tl.load(ss5+idx,mask=active,other=0.0)+dt*C23*dzvz)/azd
    a6=(axn*tl.load(ss6+idx,mask=active,other=0.0)+dt*C13*dxvx)/axd
    a7=(ayn*tl.load(ss7+idx,mask=active,other=0.0)+dt*C23*dyvy)/ayd
    a8=(azn*tl.load(ss8+idx,mask=active,other=0.0)+dt*C33*dzvz)/azd
    a9=(axn*tl.load(ss9+idx,mask=active,other=0.0)+dt*C66*dxfvy)/axd
    a10=(ayn*tl.load(ss10+idx,mask=active,other=0.0)+dt*C66*dyfvx)/ayd
    a11=(axn*tl.load(ss11+idx,mask=active,other=0.0)+dt*C55*dxfvz)/axd
    a12=(azn*tl.load(ss12+idx,mask=active,other=0.0)+dt*C55*dzfvx)/azd
    a13=(ayn*tl.load(ss13+idx,mask=active,other=0.0)+dt*C44*dyfvz)/ayd
    a14=(azn*tl.load(ss14+idx,mask=active,other=0.0)+dt*C44*dzfvy)/azd
    tl.store(ss0+idx,a0,mask=active); tl.store(ss1+idx,a1,mask=active); tl.store(ss2+idx,a2,mask=active)
    tl.store(ss3+idx,a3,mask=active); tl.store(ss4+idx,a4,mask=active); tl.store(ss5+idx,a5,mask=active)
    tl.store(ss6+idx,a6,mask=active); tl.store(ss7+idx,a7,mask=active); tl.store(ss8+idx,a8,mask=active)
    tl.store(ss9+idx,a9,mask=active); tl.store(ss10+idx,a10,mask=active); tl.store(ss11+idx,a11,mask=active); tl.store(ss12+idx,a12,mask=active); tl.store(ss13+idx,a13,mask=active); tl.store(ss14+idx,a14,mask=active)
    tl.store(sxx+idx,a0+a1+a2,mask=active); tl.store(syy+idx,a3+a4+a5,mask=active); tl.store(szz+idx,a6+a7+a8,mask=active)
    tl.store(sxy+idx,a9+a10,mask=active); tl.store(sxz+idx,a11+a12,mask=active); tl.store(syz+idx,a13+a14,mask=active)


@triton.jit
def inject_split_stress_source_kernel(source,it,idx,
                                      ss0,ss1,ss2,ss3,ss4,ss5,ss6,ss7,ss8,ss9,ss10,ss11,ss12,ss13,ss14,
                                      w0:tl.constexpr,w1:tl.constexpr,w2:tl.constexpr,w3:tl.constexpr,w4:tl.constexpr,w5:tl.constexpr):
    # Split normal-source weights into thirds and shear-source weights into halves.
    a=tl.load(source+it)
    tl.store(ss0+idx,tl.load(ss0+idx)+a*w0/3.0); tl.store(ss1+idx,tl.load(ss1+idx)+a*w0/3.0); tl.store(ss2+idx,tl.load(ss2+idx)+a*w0/3.0)
    tl.store(ss3+idx,tl.load(ss3+idx)+a*w1/3.0); tl.store(ss4+idx,tl.load(ss4+idx)+a*w1/3.0); tl.store(ss5+idx,tl.load(ss5+idx)+a*w1/3.0)
    tl.store(ss6+idx,tl.load(ss6+idx)+a*w2/3.0); tl.store(ss7+idx,tl.load(ss7+idx)+a*w2/3.0); tl.store(ss8+idx,tl.load(ss8+idx)+a*w2/3.0)
    tl.store(ss9+idx,tl.load(ss9+idx)+a*w3/2.0); tl.store(ss10+idx,tl.load(ss10+idx)+a*w3/2.0)
    tl.store(ss11+idx,tl.load(ss11+idx)+a*w4/2.0); tl.store(ss12+idx,tl.load(ss12+idx)+a*w4/2.0)
    tl.store(ss13+idx,tl.load(ss13+idx)+a*w5/2.0); tl.store(ss14+idx,tl.load(ss14+idx)+a*w5/2.0)


def _grid(n,block): return (triton.cdiv(n,block),)

def launch_pml_velocity(indices,fields,params,split_v,pml,shape,spacing,dt,coeffs,half,block):
    n=int(indices.numel());
    if n==0:return
    nz,ny,nx=shape; dx,dy,dz=spacing
    pml_velocity_fused_kernel[_grid(n,block)](indices,n,*fields,*params,*split_v,*pml,nx,ny,nz,dx,dy,dz,dt,*coeffs,HALF=half,BLOCK=block)

def launch_pml_stress(indices,fields,cij,split_s,pml,shape,spacing,dt,coeffs,half,block):
    n=int(indices.numel());
    if n==0:return
    nz,ny,nx=shape; dx,dy,dz=spacing
    pml_stress_fused_kernel[_grid(n,block)](indices,n,*fields,*cij,*split_s,*pml,nx,ny,nz,dx,dy,dz,dt,*coeffs,HALF=half,BLOCK=block)

def launch_split_source(split_s,source,it,idx,weights):
    inject_split_stress_source_kernel[(1,)](source,int(it),int(idx),*split_s,*(float(x) for x in weights),num_warps=1)
