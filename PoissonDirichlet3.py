from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve as solve
from numpy import zeros,linspace,exp,polyfit,log,sqrt,mean,array
import matplotlib.pyplot as plt
from time import time
#Main function for solving -d^2/dx^2 + d/dx = 1.0
def solveEq(nx=10,aVal=0.0,dense=False,
    secondOrder=False,dirichlet=False):
    """Solve Poisson"s equation on grid controlled by input.
    Returns x,phi,err,time"""
    #Define a function that gives the analytic solution
    #phi = c1*exp(x) + c2 + x
    if dirichlet:
        def anSol(xval,aVal):
            c2 = aVal/(1-exp(1.0))-0.5*(1+exp(1.0))/(1-exp(1.0))
            c1 = exp(0.5)*(0.5-c2)
            return c1*exp(xval)+c2+xval
    else:
        def anSol(xval,aVal):
            c2 = 0.5-(aVal-1)*exp(-1.0)
            c1 = ((aVal-1)*exp(-0.5))
            return c1*exp(xval)+c2+xval
    length = 1.0
    xval=linspace(0.0,length,num=nx)-length*0.5 ; dx = length/(nx-1.0)
    #Create matrix operator
    M = lil_matrix((nx,nx))
    #Set elements, -d^2/dx^2 +d/dx -- Skip boundaries
    rho = zeros(nx) + 1.0
    for i in range(1,nx-1):
        M[i,i-1] = -1.0/dx**2 - 0.5/dx
        M[i,i]   =  2.0/dx**2 
        M[i,i+1] = -1.0/dx**2 + 0.5/dx
    #Boundaries
    M[0,0] = 1.0 ; rho[0] = 0.0 #Always do Zero-dirichlet on left
    if dirichlet: #Dirichlet BC
        M[-1,-1] = 1.0
    else: #Neumann
        if secondOrder: #Second order
            M[-1, -1] =  1.5/(dx)
            M[-1, -2] = -2.0/(dx)
            M[-1, -3] =  0.5/(dx)
        else: #First order
            M[-1,-1] = 1.0/dx ; M[-1,-2]=-1.0/dx
    rho[-1] = aVal
    #Final setup
    M = M.tocsr()
    #Solve
    t1 = time() ; sol = solve(M,rho) ; t2 = time()
    #Calculate rms absolute error
    err=(sol-anSol(xval,aVal)); err=sqrt(mean(err*err))
    return xval,sol,err,t2-t1

nxval=array([10,20,30,35,40,45,50,100,500,1000,5000,10000])
err=[] ; tm=[]
for n in nxval:
    x,p,e,t = solveEq(nx=n,aVal=0.05,
        secondOrder=True,dirichlet=False)
    err.append(e) ; tm.append(t)
#Plot error vs nx
plt.plot(log(nxval),log(err),"-x") ; plt.xlabel(r"log$(n_x)$")
plt.ylabel(r"log$(\epsilon)$") ; plt.show()
#Fit log(error)
fit=polyfit(log(nxval),log(err),1)
order=fit[0]
print("The error scales with order : {ord}".format(ord=order))
