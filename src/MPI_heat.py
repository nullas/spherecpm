import os, petsc4py
petsc4py.init(os.sys.argv)
from petsc4py import PETSc

import scipy as sp
import scipy.sparse


def pol2car(r,tho):
    return r*sp.cos(tho),r*sp.sin(tho)

def car2pol(x,y):
    return sp.sqrt(x**2+y**2),sp.arctan2(y,x)

def ComputeCP(x):
    '''Compute CP for circle'''
    tho = car2pol(x[:,0],x[:,1])[1]
    xx,yy = pol2car(1.0,tho)
    rslt = sp.zeros((x.shape))
    rslt[:,0] = xx
    rslt[:,1] = yy
    return rslt

def GenerateDiffMatNP(k,m,n):
    kones = k*sp.ones((m*n,))
    data = sp.array([-4*kones,kones,kones,kones,kones])
    return scipy.sparse.spdiags(data, [0,-1,1,m,-m],\
                             m*n, m*n )
    
def SetProjMatPETSC(cpCorArray,ProjMat,DA,vg):
    m = DA.getSizes()[0]
    dx = 4./m
    AO = DA.getAO()
    psnx = sp.int32(sp.floor_divide(cpCorArray[:,0]+2.,dx))
    psny = sp.int32(sp.floor_divide(cpCorArray[:,1]+2.,dx))
    idxbl = psnx+psny*m
    idxbr = psnx+1+psny*m
    idxul = psnx+(psny+1)*m
    idxur = psnx+1+(psny+1)*m
    idxbl = AO.app2petsc(idxbl)
    idxbr = AO.app2petsc(idxbr)
    idxul = AO.app2petsc(idxul)
    idxur = AO.app2petsc(idxur)
    start,end = vg.getOwnershipRange()
    modx = sp.mod(cpCorArray[:,0]+2.,dx)
    mody = sp.mod(cpCorArray[:,1]+2.,dx)
    dx2 = dx**2
    for i in sp.arange(end-start):
        ProjMat[i+start,idxbl[i]] = modx[i]*mody[i]/dx2
        ProjMat[i+start,idxbr[i]] = (dx-modx[i])*mody[i]/dx2
        ProjMat[i+start,idxul[i]] = modx[i]*(dx-mody[i])/dx2
        ProjMat[i+start,idxur[i]] = (dx-modx[i])*(dx-mody[i])/dx2
    return
        

#initial
OptDB = PETSc.Options()
dimension= OptDB.getInt('dim',2)
m = OptDB.getReal('m',8)

#main
DA = PETSc.DA()
DA.create(dim = dimension,
          dof = 1,
          sizes = [m]*dimension,
          boundary_type = PETSc.DA.BoundaryType.PERIODIC,
          stencil_width=1,
          comm = PETSc.COMM_WORLD
          )

DA.setUniformCoordinates(-2.0,2.0,-2.0,2.0,-2.0,2.0)

vg = DA.createGlobalVector()
vl = DA.createLocalVector()
vCor = DA.getCoordinates()
vCorArray = vCor.getArray()
vCorArray = vCorArray.reshape((-1,dimension))
cpCorArray = ComputeCP(vCorArray)
PETSc.Sys.syncPrint(vg.getOwnershipRange(),'to',PETSc.COMM_WORLD.Get_rank())
PETSc.Sys.syncPrint(vCorArray)
PETSc.Sys.syncFlush()


#IC
vgArray = vg.getArray()
vgArray = cpCorArray[:,1]
vg.setArray(vgArray)
DA.globalToLocal(vg,vl)


dx = 4./m
dt = 0.1*dx**2
k = dt/dx**2
(ix,iy),(rx,ry) = DA.getGhostCorners()
DiffMat = GenerateDiffMatNP(k,rx,ry)
ProjMat = PETSc.Mat()
ProjMat.create()
ProjMat.setSizes((m**2,m**2))
ProjMat.setType(PETSc.Mat.Type.MPIAIJ)
ProjMat.setUp()
SetProjMatPETSC(cpCorArray,ProjMat,DA,vg)
ProjMat.assemblyBegin()
ProjMat.assemblyEnd()
vgDup = vg.duplicate()
for t in sp.arange(0,1,dt):
    DA.globalToLocal(vg,vl)
    vlArray = vl.getArray()
    vlArray = vlArray+DiffMat*vlArray
    vl.setArray(vlArray)
    DA.localToGlobal(vl,vg)
    ProjMat.mult(vg,vgDup)
    vg = vgDup.copy()
vn = DA.createNaturalVec()
DA.globalToNatural(vg,vn)
from matplotlib import pylab as pl
X,Y = sp.mgrid[-2:2:1j*m,-2:2:1j*m]
pl.figure()
Z = vg[...].reshape(m,m)
pl.contourf(X,Y,Z)
pl.axis('equal')
pl.colorbar()
pl.show()



