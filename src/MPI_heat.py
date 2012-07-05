import os, petsc4py
petsc4py.init(os.sys.argv)
from petsc4py import PETSc

import scipy as sp


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
vCor = DA.getGhostCoordinates()
vCorArray = vCor.getArray()
vCorArray = vCorArray.reshape((-1,dimension))
cpCorArray = ComputeCP(vCorArray)
PETSc.Sys.syncPrint(vg.getOwnershipRange(),'to',PETSc.COMM_WORLD.Get_rank())
PETSc.Sys.syncPrint(vCorArray)
PETSc.Sys.syncFlush()
AO = DA.getAO()
#IC
vlArray = vl.getArray()
vlArray = sp.sin(cpCorArray[:,1])
vl.setArray(vlArray)
print vl[2]
print vlArray[2]
print vlArray.size
print vl.sizes
print vg.sizes
