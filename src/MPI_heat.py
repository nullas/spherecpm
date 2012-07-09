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
    mat = scipy.sparse.lil_matrix((m*n,(m+2)*(n+2)))
    def helper_base(i,j,base,value,rm=m,rn=n,m=mat):
        m[i*rm+j,(i+1)*(rm+2)+j+1+base]=value
    def helper_1((i,j)):
        helper_base(i,j,0,-4*k)
    def helper_2((i,j)):
        helper_base(i,j,-1,k)
    def helper_3((i,j)):
        helper_base(i,j,+1,k)
    def helper_4((i,j),rm=m+2):
        helper_base(i,j,-rm,k)
    def helper_5((i,j),rm=m+2):
        helper_base(i,j,rm,k)
    li = [(i,j) for i in sp.arange(m) for j in sp.arange(n)]
    map(helper_1,li)
    map(helper_2,li)
    map(helper_3,li)
    map(helper_4,li)
    map(helper_5,li)
    return mat
        
        
    
    
def SetProjMatPETSC(cpCorArray,ProjMat,DA):
    m = DA.getSizes()[0]
    dx = 4./m

    return
        

#initial
OptDB = PETSc.Options()
dimension= OptDB.getInt('dim',2)
m = OptDB.getReal('m',40)

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
vg.setFromOptions()
vl = DA.createLocalVector()
vCor = DA.getCoordinates()
vCorArray = vCor.getArray()
vCorArray = vCorArray.reshape((-1,dimension))
cpCorArray = ComputeCP(vCorArray)

#IC

vgArray = vg.getArray()

vgArray = cpCorArray[:,1]+cpCorArray[:,0]-2

#vg.set(1)
#vtest = vg.duplicate()
#vtest.setArray(vCorArray[:,1]+vCorArray[:,0])
vg.setArray(vgArray)

DA.globalToLocal(vg,vl)
vnviewer = PETSc.Viewer().DRAW()





#code

dx = 4./m
dt = 0.1*dx**2
k = dt/dx**2
(gix,giy),(grx,gry) = DA.getGhostCorners()
(ix,iy),(rx,ry) = DA.getCorners()
PETSc.Sys.syncPrint(ix,iy,rx,ry,'to',PETSc.COMM_WORLD.Get_rank())
PETSc.Sys.syncFlush()
DiffMat = GenerateDiffMatNP(k,rx,ry)
ProjMat = DA.createMat()

#SetProjMatPETSC(cpCorArray,ProjMat,DA)
psnx = sp.int_(sp.floor_divide(cpCorArray[:,0]+2.,dx))
psny = sp.int_(sp.floor_divide(cpCorArray[:,1]+2.,dx))

modx = sp.mod(cpCorArray[:,0]+2.,dx)
mody = sp.mod(cpCorArray[:,1]+2.,dx)
dx2 = dx**2
stenRow = ProjMat.Stencil()
stenCol = ProjMat.Stencil()
index = 0
(ix,iy),(rx,ry) = DA.getCorners()
for i in range(ix,ix+rx):
    stenRow.i = i
    for j in range(iy,iy+ry):
        
        
        stenRow.j = j
        
        iCP = psnx[index]
        jCP = psny[index]
        
        iCP = 0
        jCP = 1
        
        stenCol.i = iCP
        stenCol.j = jCP
        ProjMat.setValueStencil(stenRow,
                                stenCol,
                                (dx-modx[index])*(dx-mody[index])/dx2)
        
        stenCol.i = iCP + 1
        stenCol.j = jCP
        ProjMat[1,42]=1
        ProjMat.setValueStencil(stenRow,
                                stenCol,
                                modx[index]*(dx-mody[index])/dx2)

        stenCol.i = iCP
        stenCol.j = jCP + 1
        ProjMat.setValueStencil(stenRow,
                                stenCol,
                                (dx-modx[index])*mody[index]/dx2)
        
        stenCol.i = iCP + 1
        stenCol.j = jCP + 1
        ProjMat.setValueStencil(stenRow,
                                stenCol,
                                modx[index]*mody[index]/dx2)
        index = index + 1
ProjMat.assemblyBegin()
ProjMat.assemblyEnd()


#vtest2 = vtest.copy()
#ProjMat.mult(vtest,vtest2)
#vtest2.view(viewer=vnviewer)

#PETSc.Sys.Print('Pause...')
#if PETSc.COMM_WORLD.getRank() == 0:
#    
#    raw_input()
    
PETSc.COMM_WORLD.barrier()

vgDup = vg.duplicate()
for t in sp.arange(0,1,dt):
    DA.globalToLocal(vg,vl)
    vlArray = vl.getArray()
    vgArray = vg.getArray()
    vgArray = vgArray+DiffMat*vlArray
    vg.setArray(vgArray)
    ProjMat.mult(vg,vgDup)
    vg = vgDup.copy()
#    vg.view(viewer=vnviewer)
#    PETSc.Sys.Print('Pause...')
#    if PETSc.COMM_WORLD.Get_rank() == 0:
#        raw_input()
#    PETSc.COMM_WORLD.barrier()
    

vg.view(viewer=vnviewer)
PETSc.Sys.Print('Pause...')
if PETSc.COMM_WORLD.Get_rank() == 0:
    raw_input()
PETSc.COMM_WORLD.barrier()






