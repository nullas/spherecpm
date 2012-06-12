import scipy as sp
import scipy.sparse
import scipy.linalg
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
    tempk = k*(gridsize-1)**2/16
    kones = tempk*sp.ones((gridsize**2,))
    data = sp.array([-4*kones,kones,kones,kones,kones])
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

def GenerateProjectionMat2(x,y,cpx,cpy,gridsize):
    tempcpx = cpx.reshape((-1,))
    tempcpy = cpy.reshape((-1,))
    positionx=sp.floor_divide(tempcpx+2,4./(gridsize-1))
    positiony=sp.floor_divide(tempcpy+2,4./(gridsize-1))
    mat = scipy.sparse.lil_matrix((tempcpx.size,gridsize**2))
    gridscalar = (gridsize-1)**2/16.
    for idx in range(0,tempcpx.size):
        px = int(positionx[idx])
        py = int(positiony[idx])
        mat[idx,py*gridsize+px] = gridscalar*(tempcpx[idx]-x[0,px])*(tempcpy[idx]-y[py,0])
        mat[idx,py*gridsize+px+1] = gridscalar*(x[0,px+1]-tempcpx[idx])*(tempcpy[idx]-y[py,0])
        mat[idx,(py+1)*gridsize+px] = gridscalar*(tempcpx[idx]-x[0,px])*(y[py+1,0]-tempcpy[idx])
        mat[idx,(py+1)*gridsize+px+1] = gridscalar*(x[0,px+1]-tempcpx[idx])*(y[py+1,0]-tempcpy[idx])
    return mat

if __name__=="__main__":
    #initial
    gridsizes=[40]
    theta = sp.linspace(0,2*sp.pi,100)
    xsphere = sp.cos(theta)
    ysphere = sp.sin(theta)
    
    err=[]
    pl.figure()
    
    #code
    for gridsize in gridsizes:
        k=0.001*(4./(gridsize-1))**2
        x,y=GenerateGrids(gridsize)
        cpx,cpy = ComputeCP(x,y)
        u = cpy
        u = u.reshape((-1,))
        A = GenerateDiffMat(k,gridsize)
        B = GenerateProjectionMat(x,y,cpx,cpy,gridsize)
        C = B*A
        
        for t in sp.arange(k,0.02,k):
            u = u+C*u
#            tempu = u.reshape((gridsize,-1))
#            pl.figure()
#            pl.pcolor(x,y,tempu)
            
#        pl.show()
        u_accu = sp.exp(-t)*sp.sin(theta)
        D = GenerateProjectionMat2(x,y,xsphere,ysphere,gridsize)
        u_interpolated = D*u
        err.append(scipy.linalg.norm(u_accu-u_interpolated)*2*sp.pi/99.)
        print err
        
    
    pl.loglog(gridsizes,err)
    pl.figure()
    u = u.reshape((gridsize,-1))
    pl.pcolor(x,y,u)
    pl.axis('equal')
    pl.show()
        
    
        
        
    