import scipy as sp
import pylab as pl
import scipy.linalg

def rn(m=20):
    
    rd = sp.rand(m-1)
    su = 0
    x = sp.zeros((m,))
    for i in range(m-1):
        su = su + rd[i]+1./10
        x[i+1] = su
    
    x = x/su
    return x


def GenDiffMat(x):
    size = x.size
    mat = sp.zeros((size-2,size-2))
    mat = sp.mat(mat)
    dx = x[1:]-x[:-1]
    for i in range(1,size-3):
        mat[i,i] = -2./(dx[i]*dx[i+1])
        mat[i,i-1] = 2/(dx[i]*(dx[i]+dx[i+1]))
        mat[i,i+1] = 2/(dx[i+1]*(dx[i]+dx[i+1]))
    i = 0
    mat[i,i] = -2./(dx[i]*dx[i+1])
    mat[i,i+1] = 2/(dx[i+1]*(dx[i]+dx[i+1]))
    i = size-3
    mat[i,i] = -2./(dx[i]*dx[i+1])
    mat[i,i-1] = 2/(dx[i]*(dx[i]+dx[i+1]))
    return mat
m = 20
x = rn(m)


n = 40
y = rn(n)

#x = sp.linspace(0,1,11)
mat = GenDiffMat(x)
ev,ew = scipy.linalg.eig(mat)
print ev
print '--------------------------'
print mat

#plot
pl.figure()
pl.plot(x,'ko')
pl.plot(y,'ro')
pl.figure()
pl.plot(ev.real,ev.imag,'r+')
pl.plot([-4/min(x[1:]-x[:-1])**2],[0],'ko')




pl.show()