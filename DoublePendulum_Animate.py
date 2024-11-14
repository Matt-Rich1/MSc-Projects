#Import useful routines from modules
from scipy.integrate import solve_ivp
from numpy import linspace, sin, cos, array
import matplotlib.pyplot as plt
import matplotlib.animation as ani

#Here we return the time derivatives of theta1,theta2,p1,p2
def dfdt(curT,curF,mass=1.0,length=1.0,g=9.81):
    #Unpack values
    theta1 = curF[0] ; theta2=curF[1] ; p1 = curF[2] ; p2 = curF[3]
    cDif = cos(theta1-theta2) ; sDif = sin(theta1-theta2)
    dth1 = (6/(mass*length*length))*(2*p1-3*p2*cDif)/(16-9*cDif*cDif)
    dth2 = (6/(mass*length*length))*(8*p2-3*p1*cDif)/(16-9*cDif*cDif)
    dp1 = -0.5*mass*length*length*(dth1*dth2*sDif+3*g*sin(theta1)/length)
    dp2 = 0.5*mass*length*length*(dth1*dth2*sDif-g*sin(theta2)/length)
    return [dth1,dth2,dp1,dp2]
 
#Define times, initial conditions and parameters
time=linspace(0,10,400)
theta1_0=1.5 ; theta2_0=3.5 ; f0=[theta1_0,theta2_0,0,0]
mass = 1; g = 9.81 ; length = 2
result = solve_ivp(dfdt, [time[0], time[-1]], f0,
                   t_eval = time, args=(mass,length,g))

#Plot
fig1 = plt.figure(); plt.xlabel(r"$x$"); plt.ylabel(r"$y$")
line1 = plt.plot([],[],'-') #The line connecting origin and bob1
line2 = plt.plot([],[],'go') #Pendulum bob1
line3 = plt.plot([],[],'-') #The line connecting bob1 and bob2
line4 = plt.plot([],[],'ro') #Pendulum bob2

#Calculate positions and set the plot range
plt.xlim(-2*length,2*length); plt.ylim(-2*length,2*length)
fig1.gca().set_aspect('equal')

#Function to get pendulum x-y position given theta and length
def pendPos(theta,length,origin=[0,0]):
    x = array(origin[0]+length*sin(theta)) ; y = array(origin[1]+length*cos(theta))
    return x,y

#Function to draw the pendulum
def drawPendulum(num,res,l1,l2,l3,l4,length=1,ntrail=1):
    mn=max([0,num-ntrail])
    x,y = pendPos(res[0,mn:num+1],-length)
    l1.set_data([0,x[-1]],[0,y[-1]]) ; l2.set_data(x,y)
    x2,y2 = pendPos(res[1,mn:num+1],-length,[x,y])
    l3.set_data([x[-1],x2[-1]],[y[-1],y2[-1]]) ; l4.set_data(x2,y2)    
    return l1,l2,l3,l4,
    
#Make animation and then show it
trailLen = 0# len(time) #Make this len(time) to keep all history
line_ani = ani.FuncAnimation(fig1,drawPendulum,len(time),fargs=(result.y,line1[0],
    line2[0],line3[0],line4[0],length,trailLen),interval=5,
    blit=False, repeat=False)
plt.show()
