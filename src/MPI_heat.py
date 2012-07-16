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

def GenerateDiffMat(DA,k):
    sten = PETSc.Mat.Stencil()
    sten2 = PETSc.Mat.Stencil()
    mat = DA.createMat()
    (ix,iy),(rx,ry) = DA.getCorners()
    def mat_helper((i,j),mat=mat ,sten = sten,sten2=sten2):
        
        sten.index = (i,j)
        mat.setValueStencil(sten,sten,1-4*k)
        
        sten2.index = (i,j+1)
        mat.setValueStencil(sten,sten2,k)
        
        sten2.index = (i+1,j)
        mat.setValueStencil(sten,sten2,k)
        
        sten2.index = (i-1,j)
        mat.setValueStencil(sten,sten2,k)
        
        sten2.index = (i,j-1)
        mat.setValueStencil(sten,sten2,k)
        
    map(mat_helper, [(i,j) for i in range(ix,ix+rx) for j in range(iy,iy+ry)])
    mat.assemblyBegin()
    mat.assemblyEnd()
    return mat    
        
    
    
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
        ProjMat[i+start,idxbl[i]] = (dx-modx[i])*(dx-mody[i])/dx2
        ProjMat[i+start,idxbr[i]] = modx[i]*(dx-mody[i])/dx2
        ProjMat[i+start,idxul[i]] = (dx-modx[i])*mody[i]/dx2
        ProjMat[i+start,idxur[i]] = modx[i]*mody[i]/dx2
    return
        

#initial
OptDB = PETSc.Options()
dimension= OptDB.getInt('dim',2)
m = OptDB.getReal('m',51)

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

vgArray = cpCorArray[:,1]

#vtest = vg.duplicate()
#vtest.setArray(vCorArray[:,1]+vCorArray[:,0])
vg.setArray(vgArray)

DA.globalToLocal(vg,vl)
vnviewer = PETSc.Viewer().DRAW()





#code

dx = 4./m
dt = 0.1*dx**2
k = dt/dx**2
p=3
(gix,giy),(grx,gry) = DA.getGhostCorners()
(ix,iy),(rx,ry) = DA.getCorners()
PETSc.Sys.syncPrint(ix,iy,rx,ry,'to',PETSc.COMM_WORLD.Get_rank())
PETSc.Sys.syncFlush()
DiffMat = GenerateDiffMat(DA,k)



ProjMat = PETSc.Mat().create()

ProjMat.setType(PETSc.Mat.Type.MPIAIJ)
ProjMat.setSizes((m**2,m**2))
ProjMat.setPreallocationNNZ(((p+1)**2,(p+1)**2))
ProjMat.setFromOptions()
SetProjMatPETSC(cpCorArray,ProjMat,DA,vg)
ProjMat.assemblyBegin()
ProjMat.assemblyEnd()



#test 
vtest = vg.duplicate()
vtest.setArray(vCorArray[:,1])
vtest2 = vtest.copy()
ProjMat.mult(vtest,vtest2)
vtest2.view(viewer=vnviewer)
PETSc.Sys.syncPrint(vg.getOwnershipRange())
PETSc.Sys.syncPrint(ProjMat.getOwnershipRange())
PETSc.Sys.syncFlush()
PETSc.Sys.Print('Pause...')
if PETSc.COMM_WORLD.getRank() == 0:
    
    raw_input()
    
PETSc.COMM_WORLD.barrier()

vgDup = vg.duplicate()
for t in sp.arange(0,1,dt):
    DiffMat.mult(vg,vgDup)
    ProjMat.mult(vgDup,vg)
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