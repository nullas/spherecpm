import scipy as sp
import pylab as pl
import scipy.linalg
from time import time

def rn(m=20):
    
    rd = sp.rand(m-1)
    su = 0
    x = sp.zeros((m,))
    for i in range(m-1):
        su = su + rd[i]+1./100
        x[i+1] = su
    
    x = x/su
    return x


def GenDiffMat(x):
    size = x.size
    mat = sp.zeros((size-1,size-1))
    mat = sp.mat(mat)
    dx = x[1:]-x[:-1]
    for i in range(0,size-2):
        mat[i,i] = -2./(dx[i]*dx[i-1])
        mat[i,i-1] = 2/(dx[i-1]*(dx[i-1]+dx[i]))
        mat[i,i+1] = 2/(dx[i]*(dx[i-1]+dx[i]))
    i = size - 2
    mat[i,i] = -2./(dx[i]*dx[i-1])
    mat[i,i-1] = 2/(dx[i-1]*(dx[i-1]+dx[i]))
    mat[i,0] = 2/(dx[i]*(dx[i-1]+dx[i]))
    return mat

def Weights(yCor,x,ind):
    ww = sp.zeros(4)
    base = 1
    extx = sp.concatenate(([x[-2]-1],x,[x[1]+1]))
    if yCor == x[ind]:
        ww[2] = 1
    else:
        for k in range(0,4):
            poly = sp.poly1d(sp.poly(extx[[i for i in range(base+ind-2,base+ind+2) if i != base+ind+k-2]]))
            ww[k] = poly(yCor)/poly(extx[base+ind+k-2])
        
    return ww

def GenExtensionMat(y,x):
    m = y.size-1
    n = x.size-1
    mat = sp.zeros((m,n))
    mat = sp.mat(mat)
    indBase = sp.searchsorted(x, y)
    for i in range(0,m):
        ww = Weights(y[i],x,indBase[i])
        try:
            mat[i,indBase[i]-2:indBase[i]+2] = ww
        except(IndexError,ValueError):
            mat[i,(indBase[i]-2)%n] = ww[0]
            mat[i,(indBase[i]-1)%n] = ww[1]
            mat[i,(indBase[i])%n] = ww[2]
            mat[i,(indBase[i]+1)%n] = ww[3]
    return mat
    
    
    



n = 60
y = rn(n)

m = 50

xind = sp.random.randint(0,n-1,m)
sp.concatenate((xind,[n-1]))
xind = sp.unique(xind)

x = y[xind]


#x = sp.linspace(0,1,101)
mat = GenDiffMat(y)
ev,ew = scipy.linalg.eig(mat)
idxsort = sp.argsort(ev)
print sp.sort(ev)
print '-------------------------------------\n'*3


ExtMat = GenExtensionMat(y,x)
#fx = sp.sin(sp.pi*2*x)[:-1]
#tic = time()
#fx = fx.reshape((-1,1))
#print 'reshape takes\n'
#print time()-tic
#print '----------------------------------------\n'
#fy = ExtMat*fx


Mat = mat[xind[:-1],:]*ExtMat
ev2,ew2 = scipy.linalg.eig(Mat)
print sp.sort(ev2)
#plot
pl.figure()
pl.plot(ev2.real,ev2.imag,'r+')
pl.title('eigenvalues of E*L*E')
pl.figure()
pl.plot(ev.real,ev.imag,'r+')
pl.plot([-4/min((y[2:]-y[1:-1])*(y[1:-1]-y[:-2]))],[0],'ko')
pl.title('Eigenvalues of normal Laplacian on nonuniform grid')
#pl.figure()
#pl.plot(x[0:-1],ew[:,0],label='0')
#pl.plot(x[0:-1],ew[:,idxsort[-1]],label='max')
#pl.plot(x[0:-1],ew[:,idxsort[0]],label='min')
#pl.plot(x[0:-1],ew[:,idxsort[-2]],label='second min')
#pl.plot(x[0:-1],ew[:,idxsort[-20]],label='3rd min')
#pl.legend(loc='best')
#pl.figure()
#pl.plot(x[:-1],fx)
#pl.plot(y[:-1],fy)



pl.show()