"""Inject 3-D point moment-tensor sources and sample receiver velocities.

Stress weights follow [sxx, syy, szz, sxy, sxz, syz]; output records
are laid out as [time, receiver].
"""

from __future__ import annotations
import triton
import triton.language as tl


@triton.jit
def inject_source_kernel(sxx,syy,szz,sxy,sxz,syz,source,it,idx,
                         w0: tl.constexpr,w1: tl.constexpr,w2: tl.constexpr,
                         w3: tl.constexpr,w4: tl.constexpr,w5: tl.constexpr):
    """Inject a single-node moment-tensor source; runtime it/idx avoid per-step JIT specialization."""
    # Add this time sample at the flattened source index in stress-component order.
    amp=tl.load(source+it)
    tl.store(sxx+idx,tl.load(sxx+idx)+amp*w0)
    tl.store(syy+idx,tl.load(syy+idx)+amp*w1)
    tl.store(szz+idx,tl.load(szz+idx)+amp*w2)
    tl.store(sxy+idx,tl.load(sxy+idx)+amp*w3)
    tl.store(sxz+idx,tl.load(sxz+idx)+amp*w4)
    tl.store(syz+idx,tl.load(syz+idx)+amp*w5)


@triton.jit
def sample_kernel(vx,vy,vz,rec_idx,outx,outy,outz,row,nrec: tl.constexpr,BLOCK: tl.constexpr):
    # Gather velocity samples into one row of each [time, receiver] output array.
    i=tl.arange(0,BLOCK)
    m=i<nrec
    idx=tl.load(rec_idx+i,mask=m,other=0)
    tl.store(outx+row*nrec+i,tl.load(vx+idx,mask=m,other=0.0),mask=m)
    tl.store(outy+row*nrec+i,tl.load(vy+idx,mask=m,other=0.0),mask=m)
    tl.store(outz+row*nrec+i,tl.load(vz+idx,mask=m,other=0.0),mask=m)


def launch_source(stress_fields,source,it,idx,weights):
    inject_source_kernel[(1,)](*stress_fields,source,int(it),int(idx),*(float(x) for x in weights),num_warps=1)


def launch_sample(vel_fields,rec_idx,outs,row):
    n=int(rec_idx.numel())
    block=triton.next_power_of_2(n)
    sample_kernel[(1,)](*vel_fields,rec_idx,*outs,int(row),nrec=n,BLOCK=block,num_warps=4)
