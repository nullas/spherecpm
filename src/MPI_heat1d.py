import os, petsc4py
petsc4py.init(os.sys.argv)
from petsc4py import PETSc
import numpy

import scipy as sp

from matplotlib import pylab as pl

OptDB = PETSc.Options()

mlist = sp.array([50,100,200])
error = sp.array([])
for m in mlist:
    
    
    DA = PETSc.DA().create(dim = 1,
                           sizes = (m,),
                           boundary_type = PETSc.DA.BoundaryType.PERIODIC,
                           stencil_width=1,
                           comm = PETSc.COMM_WORLD
                           )
    DA.setUniformCoordinates(-1,1)
    
    vg = DA.createGlobalVec()
    
    
    #IC u_0 = sin(x) 
    vgCorArray = DA.getCoordinates().getArray()
    vg.setArray(sp.sin(sp.pi*vgCorArray))
    
    
    #Generate Laplacian
    
    L = DA.createMat()
    #L.setUp()
    
    dx =  2./m
    dx2 = dx**2
    dt = 0.1*dx2
    (ix,),(rx,) = DA.getCorners()
    stenRow = L.Stencil()
    stenLeft = L.Stencil()
    stenRight = L.Stencil()
    k = dt/dx2
    print L.getOwnershipRange()
    for i in range(ix,ix+rx):
        stenRow.i = i
        stenLeft.i = i-1
        stenRight.i = i+1
        L.setValueStencil(stenRow,stenRow,1+2*k)
        L.setValueStencil(stenRow,stenLeft,-k)
        L.setValueStencil(stenRow,stenRight,-k)
    L.assemblyBegin()
    L.assemblyEnd()
    L.view()
    
    KSP = PETSc.KSP().create()
    KSP.setOperators(L)
    KSP.setType('cg')
    PC = KSP.getPC()
    PC.setType('none')
    KSP.setFromOptions()
    
    vnew = vg.duplicate()
    
    for t in sp.arange(0,1,dt):
        KSP.solve(vg,vnew)   # solve linear system L vnew = vg
        vg = vnew.copy()
        
    if PETSc.COMM_WORLD.getRank() == 0:
        
        u = sp.array([vg[numpy.floor(m/4)]])
        u_acc = sp.array([sp.exp(-t*sp.pi**2)*sp.sin(sp.pi*(DA.getCoordinates()[2]))])
        error = sp.concatenate((error,sp.absolute(u-u_acc)))
        
    
    
        
vgview = PETSc.Viewer.DRAW()
vg.view(viewer = vgview)
vg.view()

if PETSc.COMM_WORLD.getRank() == 0:
    pl.loglog(mlist,error)
    pl.show()

    
PETSc.Sys.Print('Pause')
if PETSc.COMM_WORLD.getRank() == 0:
    raw_input()
PETSc.COMM_WORLD.barrier()

    
