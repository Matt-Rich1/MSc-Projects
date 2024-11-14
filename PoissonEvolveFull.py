from scipy.fftpack import fft, fftfreq, ifft, fftshift
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numpy import linspace, sin, pi, sqrt, mean, zeros, concatenate

def rhoToPhi(x,rho):
    """Takes the x grid and rho values and returns
        phi using an FFT integration approach
    """
    #Get properties of x-grid
    nx = len(x) ; dx = x[1]-x[0]
    #Now FFT to get rhoHat
    rhoHat = fft(rho)
    #Get the inverse wavelengths
    lambInv = fftfreq(nx,dx)
    #Calculate wavenumbers
    kx = lambInv*2*pi
    #Now we can find phiHat using phiHat = rhoHat / kx^2 epsilon0
    #Note: we've normalised such that epsilon0==1.0
    phiHat = zeros(len(rhoHat),dtype='complex')
    phiHat[1:] = rhoHat[1:] / kx[1:]**2
    phi = ifft(phiHat)
    return phi.real
    
def timeDeriv(state,time,x):
    """ Takes the state [rho,drho/dt=v] and returns
        [drho/dt,d^2rho/dt^2]. Also requires x to be passed
    """
    #Work out length of x
    nx = len(x)
    #Unpack
    rho = state[:nx]
    v = state[nx:]
    
    #Calculate the electrostatic potential
    phi = rhoToPhi(x,rho)   
    #Return the time derivatives
    return concatenate([v,-phi])

#Setup the problem
nx = 50 ; length = 1.0 
x = linspace(0.0,length,nx,endpoint=False)
dx = x[1]-x[0]

#Determine the initial conditions
rho0 = sin(2*pi*x/length)+sin(2*pi*5*x/length)
drhodt0 = 0.0*rho0
initVal = concatenate([rho0,drhodt0])
#Times for output
t = linspace(0,200.0,100)
#Get result
result = odeint(timeDeriv,initVal,t,args=(x,))

#Unpack result
rho = result[:,:nx] #All times and the first nx points
#Make a contour plot
plt.contourf(x,t,rho,64)
plt.xlabel(r'$x$') ; plt.ylabel(r'$t$')
plt.colorbar(label=r'$\rho$')
plt.show()
#Look at FFT coefficients
rhoHat = (1.0/nx)*(fft(rho,axis=1).imag)
invLamb = fftfreq(nx,dx)
plt.figure(2)
plt.contourf(fftshift(invLamb),t,fftshift(rhoHat,axes=1),64)
plt.xlabel(r'$1/\lambda$') ; plt.ylabel(r'$t$')
plt.colorbar(label=r'$\hat{\rho}$')
plt.show()
