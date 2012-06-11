import scipy as sp
import scipy.sparse
import pylab as pl


def pol2cor(r,tho):
    return r*sp.cos(tho),r*sp.sin(tho)

def cor2pol(x,y):
    return sp.sqrt(x**2+y**2),sp.arctan2(y,x)

def GenerateGrids(gsize):
    x = sp.linspace(-2,2,gridsize)
    y = sp.linspace(-2,2,gridsize)
    return sp.meshgrid(x, y)
    
    
def GenerateDiffMat(k,gridsize):
    kones = k*sp.ones((gridsize**2,))
    minuskones = -1*kones
    data = sp.array([4*kones,minuskones,minuskones,minuskones,minuskones])
    return scipy.sparse.spdiags(data, [0,-1,1,gridsize,-gridsize],\
                             gridsize**2, gridsize**2 )

def ComputeCP(x,y):
    temp = sp.ones(x.shape)
    return pol2cor(temp,cor2pol(x,y)[1])


def GenerateProjectionMat(x,y,cpx,cpy,gridsize):
    positionx=sp.floor_divide(cpx+2,4./(gridsize-1))
    positiony=sp.floor_divide(cpy+2,4./(gridsize-1))
    mat = scipy.sparse.lil_matrix((gridsize**2,gridsize**2))
    gridscalar = (gridsize-1)**2/16.
    for idxy in range(0,gridsize):
        tempidxy = idxy*gridsize
        for idxx in range(0,gridsize):
            px = int(positionx[idxy,idxx])
            py = int(positiony[idxy,idxx])
            mat[tempidxy+idxx,py*gridsize+px] = gridscalar*(cpx[idxy,idxx]-x[0,px])*(cpy[idxy,idxx]-y[py,0])
            mat[tempidxy+idxx,py*gridsize+px+1] = gridscalar*(x[0,px+1]-cpx[idxy,idxx])*(cpy[idxy,idxx]-y[py,0])
            mat[tempidxy+idxx,(py+1)*gridsize+px] = gridscalar*(cpx[idxy,idxx]-x[0,px])*(y[py+1,0]-cpy[idxy,idxx])
            mat[tempidxy+idxx,(py+1)*gridsize+px+1] = gridscalar*(x[0,px+1]-cpx[idxy,idxx])*(y[py+1,0]-cpy[idxy,idxx])
    return mat

if __name__=="__main__":
    #initial
    gridsize=30
    k=0.05*(4./gridsize)**2
    err=[]
    
    #code
    x,y=GenerateGrids(gridsize)
    cpx,cpy = ComputeCP(x,y)
    u = cpy
    pl.figure()
    pl.pcolor(x,y,u)
    pl.show()
    u = u.reshape((-1,))
    A = GenerateDiffMat(k,gridsize)
    B = GenerateProjectionMat(x,y,cpx,cpy,gridsize)
    C = B*A
    
    for t in sp.arange(0,0.1,k):
        u = u+C*u
    u = u.reshape((gridsize,-1))
    pl.figure()
    pl.pcolormesh(x,y,u)
    pl.show()
    
        
        
    