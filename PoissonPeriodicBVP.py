#Imports
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy import zeros,linspace,abs,pi,sin,mean,sqrt
import matplotlib.pyplot as plt
from time import time

#Parameters
length = 1.0 ; nx = 1000
#Note we use nx-1 most places as we're skipping the repeated
#periodic point
xval=linspace(0,length,num=nx-1,endpoint=False)
dx = xval[1]-xval[0]

#Create matrix operator
M = lil_matrix((nx-1,nx-1))

#Set elements, -d^2/dx^2
rho = sin(xval*2*pi)
for i in range(0,nx-1):
    #Periodic indices
    im = i-1
    ip = (i+1) % (nx-1)
    M[i,im] = -1.0/dx**2
    M[i,i]   =  2.0/dx**2
    M[i,ip] = -1.0/dx**2

#Final setup and get solve routine
M = M.tocsr()
#Solve (and time)
t1 = time() ; phi = spsolve(M,rho) ; t2 = time()
#Note we can add any constant to our solution phi without
#changing the solution, hence we'll make phi[0] = 0.0
phi = phi-phi[0]

#Analytic solution
phiAn = sin(2*pi*xval)/(2*pi)**2
err = phiAn-phi
rmsErr = sqrt(mean(err*err.conjugate()))
print("The rms error is {err}".format(err=rmsErr))

#Plot solution
plt.plot(xval,phi,'-',label='Periodic BVP') 
plt.plot(xval[::10],phiAn[::10],'x',label='Analytic solution')
plt.xlabel(r'$x$') ; plt.ylabel(r'$\phi$')
plt.legend(loc='best') ; plt.show()
