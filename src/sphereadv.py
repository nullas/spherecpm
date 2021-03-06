import scipy as sp
import scipy.sparse
import scipy.linalg
import scipy.interpolate
import pylab as pl


def pol2cor(r,tho):
    return r*sp.cos(tho),r*sp.sin(tho)

def cor2pol(x,y):
    return sp.sqrt(x**2+y**2),sp.arctan2(y,x)

def GenerateGrids(gsize):
    x = sp.linspace(-2,2,gridsize)
    y = sp.linspace(-2,2,gridsize)
    return sp.meshgrid(x, y)
    
    
def GenerateDiffMat(x,y,k,gridsize):
    mat = scipy.sparse.lil_matrix((gridsize**2,gridsize**2))
    r,tho = cor2pol(x,y)
    w1 = -r*sp.sin(tho)
    w2 = r*sp.cos(tho)
    for idxy in range(1,gridsize-1):
        tempidxy = idxy*gridsize
        for idxx in range(1,gridsize-1):
            mat[tempidxy+idxx,tempidxy+idxx+1] = w1[idxy,idxx]
            mat[tempidxy+idxx,tempidxy+idxx-1] = -w1[idxy,idxx]
            mat[tempidxy+idxx,tempidxy+gridsize+idxx] = w2[idxy,idxx]
            mat[tempidxy+idxx,tempidxy-gridsize+idxx] = -w2[idxy,idxx]
    tempk = k/(2.*(4./(gridsize-1)))
    return tempk*mat
        
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
    gridsizes=[20,70,140]
    theta = sp.linspace(0,2*sp.pi,100)
    xsphere = sp.cos(theta)
    ysphere = sp.sin(theta)
    
    err=[]
    
    #code
    for gridsize in gridsizes:
        k=0.01*04./(gridsize-1)
        x,y=GenerateGrids(gridsize)
        cpx,cpy = ComputeCP(x,y)
        u = cpy
        u = u.reshape((-1,))
        A = GenerateDiffMat(x,y,k,gridsize)
        cpx_reshaped = cpx.reshape((-1,))
        cpy_reshaped = cpy.reshape((-1,))
        x_reshaped = sp.linspace(-2,2,gridsize)
        y_reshaped = sp.linspace(-2,2,gridsize)
        
        for t in sp.arange(k,0.5,k):
            u = u+A*u
            tempu = u.reshape((gridsize,-1))
            f = scipy.interpolate.RectBivariateSpline(x_reshaped,y_reshaped,tempu)
            u = f.ev(cpy_reshaped,cpx_reshaped)
            
        u_accu = sp.sin(theta+t)
        u = u.reshape((gridsize,-1))
        f = scipy.interpolate.RectBivariateSpline(x_reshaped,y_reshaped,u)
        u_interpolated = f.ev(ysphere,xsphere)
        err.append(scipy.linalg.norm(u_accu-u_interpolated)*2*sp.pi/99.)
        print err
        
    pl.figure()
    pl.loglog(gridsizes,err)
    pl.figure()
    pl.pcolor(x,y,u)
    pl.colorbar()
    pl.axis('equal')
    pl.show()
        
    
        
        
    