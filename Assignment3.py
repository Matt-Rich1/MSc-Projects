#ASSIGNMENT 3 - Modified Plasma Sheath.
#IMPORTS

from numpy import zeros, sqrt, concatenate, linspace, pi, exp, arange, interp, reshape, array
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#End of Imports
#------------------------------------------------------------------

#Functions
def getF(phi,vs):
    N=len(phi) ; F = zeros(N)
    #F = ni_hat - ne_hat
    ni_hat = (vs/(sqrt(vs**2 - 2*phi)))
    ne_hat = (exp(phi))
    F =  ni_hat - ne_hat #Formula of double differential, d2phi/dx2
    return F

def getV(vi, E, L):
    N = len(vi) ; V = zeros(N)
    #dv/dx = E/v - v/L
    V = E/vi - vi/L
    return V

def f(pos, state, L):   
    #Work out N from state
    N = len(state) // 2

#state is length 3 and phi/E/vi have a single value at a given position
#so phi = state[0] ; E = state[1] ; vi = state[2]
    #Unpack variables
    phi = state[0:N] 
    E = state[(N):2*N]
    vi = state[(2*N):(3*N)]

    F = getF(phi,vs)
    V = getV(vi, E, L)
    #Note dphi/dx=-E, dE/dx= F, dv/dx = V
    return concatenate([ -E, F, V ])



#End of Functions
#------------------------------------------------------------------
#INITIAL VALUES
#Iterating L*10**n
n = arange(-1,5)
l=len(n)
L_values=[]
for k in range(l):
    L_values.append(10**(float(n[k]))) #iterates values of L=0.1, 1, 10,..., 10**4


vs = 2.0 
#[phi,E, vi], where vi0 = vs
initial = [0.0, 0.001, 2.0]
mr = 1840 #mass ration of mi/me

#Constant -> sqrt(mi/me * 1/2*pi)
A = sqrt(mr * 1/(2*pi))

#start of zero, run to 40, in 100 steps
x = linspace(0, 40, 100)

#creation of variables used inside loops
xw = []
j=()
vw=[] #velocity at the wall

#creating main subplot for inset
fig, ax = plt.subplots()
axins = inset_axes(ax, width='25%', height='50%', loc='upper center')#[x start, y start, length, height]

for L in L_values:
    result = solve_ivp(lambda pos, state: f(pos, state, L), [x[0], x[-1]], initial, t_eval=x)
    #Using the lambda function to wrap in L to the solve_ivp function f(pos, state, L)
    
    phi_hat = result.y[0,:]
    vi_hat = result.y[2,:]

    j=A*exp(phi_hat) -1 
    
#Finding the wall
    xw = interp(0, -j, x)#(y_value, y(x), x); negative to have an increasing function for interp
#finding the velocity at the wall
    vw = interp(xw, x, vi_hat)#(x_value, x, y(x))

    ax.plot( x-xw,vi_hat, '-o', label='$vi(x), L= {}$'.format(L),linewidth=1.0, markersize=2.0)
    axins.plot(L, vw, '-x', markersize=5.0)

#End of Task 2
#------------------------------------------------------------------

ax.set_title('''Normalised ion velocity, v\u1d62,
as a function of the distance into the sheath, x-xw
for differnet normalised values of collision distance L''', fontsize=12)
ax.set_xlabel(r'Debye Lengths from the wall [$x - xw/ \lambda$]')
ax.set_ylabel(u'Normalised ion velocity [$v\u1d62 / c\u209B$]')
ax.grid(True)
ax.set_xticks(arange(- 10, max(x)-xw, 5.0))
ax.set_yticks(arange(0, max(vi_hat)+1,1.0))
ax.legend(loc='upper right')
axins.set_xscale('log')
axins.set(xlim=(1e-1, 1e4), ylim=(0,4))
axins.set_xlabel(r'''collision length, L,
normalised to the debye length''')
axins.set_ylabel(u'[$v\u1d62 / c\u209B$] at the wall')

axins.set_xticks([1, 1e+2, 1e+4] )
axins.get_xaxis().set_major_formatter(ScalarFormatter())
axins.grid(True)


#plt.savefig('Assignment3fig.png')
plt.show()

#End of Task 3
#End Plotting
#-------------------------------------------------------------------
