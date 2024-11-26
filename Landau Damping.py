#!/usr/bin/env python3
# Electrostatic PIC code in a 1D cyclic domain

from numpy import arange, concatenate, zeros, linspace, floor, array, pi, savetxt, column_stack, loadtxt
from numpy import sin, cos, sqrt, random, histogram, abs, sqrt, max, exp, interp, polyfit, poly1d, log10, mean, diff

import matplotlib.pyplot as plt # Matplotlib plotting library

from scipy.interpolate import interp1d

from scipy.optimize import curve_fit

import time

from scipy.signal import find_peaks

try:
    import matplotlib.gridspec as gridspec  # For plot layout grid
    got_gridspec = True
except:
    got_gridspec = False

try:                        # Need an FFT routine, either from SciPy or NumPy
    from scipy.fftpack import fft, ifft
except:                     # No SciPy FFT routine. Import NumPy routine instead
    from numpy.fft import fft, ifft

#End of Imports
#*****************************************************************************

def rk4step(f, y0, dt, args=()):
    """ Takes a single step using RK4 method """
    k1 = f(y0, *args)
    k2 = f(y0 + 0.5*dt*k1, *args)
    k3 = f(y0 + 0.5*dt*k2, *args)
    k4 = f(y0 + dt*k3, *args)

    return y0 + (k1 + 2.*k2 + 2.*k3 + k4)*dt / 6.

def calc_density(position, ncells, L):
    """ Calculate charge density given particle positions
    
    Input
      position  - Array of positions, one for each particle
                  assumed to be between 0 and L
      ncells    - Number of cells
      L         - Length of the domain

    Output
      density   - contains 1 if evenly distributed
    """
    ncells = int(ncells)
    density = zeros([ncells])
    nparticles = len(position)
    
    dx = L / ncells       # Uniform cell spacing
    for p in position / dx:    # Loop over all the particles, converting position into a cell number
        plower = int(p)        # Cell to the left (rounding down)
        offset = p - plower    # Offset from the left
        density[plower] += 1. - offset 
        density[(plower + 1) % ncells] += offset 
    # nparticles now distributed amongst ncells
    density *= float(ncells) / float(nparticles)  # Make average density equal to 1
    return density

def periodic_interp(y, x):
    """
    Linear interpolation of a periodic array y at index x
    
    Input

    y - Array of values to be interpolated
    x - Index where result required. Can be an array of values
    
    Output
    
    y[x] with non-integer x
    """
    ny = len(y)
    if len(x) > 1:
        y = array(y) # Make sure it's a NumPy array for array indexing
    xl = floor(x).astype(int) # Left index
    dx = x - xl
    xl = ((xl % ny) + ny) % ny  # Ensures between 0 and ny-1 inclusive
    return y[xl]*(1. - dx) + y[(xl+1)%ny]*dx

def fft_integrate(y):
    """ Integrate a periodic function using FFTs
    """
    n = len(y) # Get the length of y
    
    f = fft(y) # Take FFT
    # Result is in standard layout with positive frequencies first then negative
    # n even: [ f(0), f(1), ... f(n/2), f(1-n/2) ... f(-1) ]
    # n odd:  [ f(0), f(1), ... f((n-1)/2), f(-(n-1)/2) ... f(-1) ]
    
    if n % 2 == 0: # If an even number of points
        k = concatenate( (arange(0, n/2+1), arange(1-n/2, 0)) )
    else:
        k = concatenate( (arange(0, (n-1)/2+1), arange( -(n-1)/2, 0)) )
    k = 2.*pi*k/n
    
    # Modify frequencies by dividing by ik
    f[1:] /= (1j * k[1:]) 
    f[0] = 0. # Set the arbitrary zero-frequency term to zero
    
    return ifft(f).real # Reverse Fourier Transform
   

def pic(f, ncells, L):
    """ f contains the position and velocity of all particles
    """
    nparticles = len(f) // 2     # Two values for each particle
    pos = f[0:nparticles] # Position of each particle
    vel = f[nparticles:]      # Velocity of each particle

    dx = L / float(ncells)    # Cell spacing

    # Ensure that pos is between 0 and L
    pos = ((pos % L) + L) % L
    
    # Calculate number density, normalised so 1 when uniform
    density = calc_density(pos, ncells, L)
    
    # Subtract ion density to get total charge density
    rho = density - 1. ##Maybe needs changed##
    
    # Calculate electric field
    E = -fft_integrate(rho)*dx
    
    # Interpolate E field at particle locations
    accel = -periodic_interp(E, pos/dx)

    # Put back into a single array
    return concatenate( (vel, accel) )

####################################################################

def run(pos, vel, L, ncells=None, out=[], output_times=linspace(0,20,100), cfl=0.5):
    
    if ncells == None:
        ncells = int(sqrt(len(pos))) # A sensible default

    dx = L / float(ncells)
    
    f = concatenate( (pos, vel) )   # Starting state
    nparticles = len(pos)
    
    time = 0.0
    for tnext in output_times:
        # Advance to tnext
        stepping = True
        while stepping:
            # Maximum distance a particle can move is one cell
            dt = cfl * dx / max(abs(vel))
            if time + dt >= tnext:
                # Next time will hit or exceed required output time
                stepping = False
                dt = tnext - time
            f = rk4step(pic, f, dt, args=(ncells, L))
            time += dt
            
        # Extract position and velocities
        pos = ((f[0:nparticles] % L) + L) % L
        vel = f[nparticles:]
        
        # Send to output functions
        for func in out:
            func(pos, vel, ncells, L, time)
        
    return pos, vel

####################################################################
# 
# Output functions and classes
#

class Plot:
    """
    Displays three plots: phase space, charge density, and velocity distribution
    """
    def __init__(self, pos, vel, ncells, L):
        d = calc_density(pos, ncells, L)
        vhist, bins  = histogram(vel, int(sqrt(len(vel))))
        vbins = 0.5*(bins[1:]+bins[:-1])
        
        # Plot initial positions
        if got_gridspec:
            self.fig = plt.figure()
            self.gs = gridspec.GridSpec(4, 4)
            ax = self.fig.add_subplot(self.gs[0:3,0:3])
            self.phase_plot = ax.plot(pos, vel, '.')[0]
            ax.set_title("Phase space")
            
            ax = self.fig.add_subplot(self.gs[3,0:3])
            self.density_plot = ax.plot(linspace(0, L, ncells), d)[0]
            
            ax = self.fig.add_subplot(self.gs[0:3,3])
            self.vel_plot = ax.plot(vhist, vbins)[0]
        else:
            self.fig = plt.figure()
            self.phase_plot = plt.plot(pos, vel, '.')[0]
            
            self.fig = plt.figure()
            self.density_plot = plt.plot(linspace(0, L, ncells), d)[0]
            
            self.fig = plt.figure()
            self.vel_plot = plt.plot(vhist, vbins)[0]
        plt.ion()
        plt.show()
        
    def __call__(self, pos, vel, ncells, L, t):
        d = calc_density(pos, ncells, L)
        vhist, bins  = histogram(vel, int(sqrt(len(vel))))
        vbins = 0.5*(bins[1:]+bins[:-1])
        
        self.phase_plot.set_data(pos, vel) # Update the plot
        self.density_plot.set_data(linspace(0, L, ncells), d)
        self.vel_plot.set_data(vhist, vbins)
        plt.draw()
        plt.pause(0.05)
        

class Summary:
    def __init__(self):
        self.t = []
        self.firstharmonic = []
        
    def __call__(self, pos, vel, ncells, L, t):
        # Calculate the charge density
        d = calc_density(pos, ncells, L)
        
        # Amplitude of the first harmonic
        fh = 2.*abs(fft(d)[1]) / float(ncells)
        
        self.t.append(t)
        self.firstharmonic.append(fh)

    def save_to_file(self):
        """Save time and first harmonic amplitude to a file."""
        data = column_stack((self.t, self.firstharmonic))  # Combine into a 2D array
        header = "Time First_Harmonic_Amplitude"  
        savetxt('firstharmonic.txt', data, header=header, fmt="%.6e")

    def extract_peaks(self):
    # Load the data from the file
        data = loadtxt('firstharmonic.txt')
    
    # Split data into time and amplitude
        times = data[:, 0]
        amplitudes = data[:, 1]
    
    # Find peaks in the amplitude data
        peak_indices, _ = find_peaks(amplitudes)
    
    # Extract times and amplitudes of peaks
        self.peak_times = times[peak_indices]
        self.peak_amplitudes = amplitudes[peak_indices]
        

        for i in range(1, len(self.peak_amplitudes)):
            if self.peak_amplitudes[i] > self.peak_amplitudes[i - 1]:
                self.first_increasing_time = self.peak_times[i]
                self.first_increasing_amplitude = self.peak_amplitudes[i]
                break #stop after finding the first peak
 
    def peak_amplitude(self):
        data = loadtxt('firstharmonic.txt')
    
    # Split data into time and amplitude
        t = data[:, 0]
        A = data[:, 1]

        peaks_indices, _ = find_peaks(A)
        peaks=[]
        times=[]

        for i in peaks_indices:
            peaks.append(A[i])
            times.append(t[i])

        peaks_times = t[peaks_indices]

        return peaks, times, peaks_times
        


    def freq_calc(self):
        peaks = self.peak_amplitude()[0]
        times = self.peak_amplitude()[1]
        peaks_times = self.peak_amplitude()[2]

# Compute time differences between consecutive peaks
        time_gaps = diff(peaks_times)

# Calculate the average time gap
        average_time_gap = mean(time_gaps)
        max_time_gap = max(time_gaps)
        min_time_gap = min(time_gaps)
        freq = 1 / average_time_gap
        
        error_freq =  (max_time_gap - min_time_gap)/2
        percentage_error_freq = abs((1 - 0.5*error_freq**2) / (1 - 0.5*average_time_gap**2))

        #plt.errorbar(ncells, freq, freq*percentage_error_freq)

        return freq, percentage_error_freq


    def dr_calc(self):
        data = loadtxt('firstharmonic.txt')
    
    # Split data into time and amplitude
        t = data[:, 0]
        A = data[:, 1]

        peaks = self.peak_amplitude()[0]
        times = self.peak_amplitude()[1]
        
        threshold = A[0]/2.71828282
        tau = ()
        error_dr = ()
# Create an interpolation function for peaks over times
        interp_func = interp1d(peaks, times, kind='linear', bounds_error=False, fill_value='extrapolate')

# Find the time where peaks equals the threshold
        tau = interp_func(threshold)
        
# Calculate error (absolute difference between threshold and interpolated value at tau)
        error_dr = abs(threshold - interp_func(tau))/interp_func(tau)
                

        dr = 1 / tau
        percentage_error_dr = abs((1 + error_dr + 0.5*error_dr**2) / (1 + tau + 0.5*tau**2))

       #plt.errorbar(ncells, dr, dr*percentage_error_dr)

        return dr, percentage_error_dr


    def noise_calc(self):

        data = loadtxt('firstharmonic.txt')
    
    # Split data into time and amplitude
        t = data[:, 0]
        A = data[:, 1]
            # Split the data into signal and noise

        peaks = self.peak_amplitude()[0]
            
#data = noise+signal
        Amp = A[0]; dr = self.dr_calc()[0]; freq = self.freq_calc()[0]; w = 2*3.14159 * freq
        signal = Amp * (exp(-(dr*t))) * cos(w*t)

        residuals = signal - A
        residuals_squared = residuals**2

        noise = sqrt(mean(residuals_squared))

        f = self.freq_calc()[1]
        d = self.dr_calc()[1]

        error = sqrt(d**2 + f**2)

        #plt.errorbar(ncells, noise, noise*error)

        return noise


####################################################################
# Functions to create the initial conditions
#

def landau(npart, L, alpha=0.2):
    """
    Creates the initial conditions for Landau damping
    
    """
    # Start with a uniform distribution of positions
    pos = random.uniform(0., L, npart)
    pos0 = pos.copy()
    k = 2.*pi / L
    for i in range(10): # Adjust distribution using Newton iterations
        pos -= ( pos + alpha*sin(k*pos)/k - pos0 ) / ( 1. + alpha*cos(k*pos) )
        
    # Normal velocity distribution
    vel = random.normal(0.0, 1.0, npart)
    
    return pos, vel

def twostream(npart, L, vbeam=2):
    # Start with a uniform distribution of positions
    pos = random.uniform(0., L, npart)
    # Normal velocity distribution
    vel = random.normal(0.0, 1.0, npart)
    
    np2 = int(npart / 2)
    vel[:np2] += vbeam  # Half the particles moving one way
    vel[np2:] -= vbeam  # and half the other
    
    return pos,vel

####################################################################

if __name__ == "__main__":
    # Generate initial condition
    #
    npart = 9000
    ncells = 18
    L = 72
    freq = []
    dr = []
    noise = []
    start_time = time.time()
    
    if True:
        # 2-stream instability
        L = int(L)
        pos, vel = twostream(npart, L, 3.) # Might require more npart than Landau!
    else:
        # Landau damping
        L = float(L*pi/100)
        pos, vel = landau(npart, L)
    
    # Create some output classes
    p = Plot(pos, vel, ncells, L) # This displays an animated figure - Slow!
    s = Summary()                 # Calculates, stores and prints summary info

    diagnostics_to_run = [p, s]   # Remove p to get much faster code!

    
    # Run the simulation

    pos, vel = run(pos, vel, L, ncells, 
                   out = diagnostics_to_run,        # These are called each output step
                    output_times=linspace(0.,20,50)) # The times to output

    s.save_to_file()  # Save data to file
    s.extract_peaks()  # Find peaks and the first increasing peak
    s.freq_calc()
    s.dr_calc()
    s.noise_calc()

    freq_value = s.freq_calc()[0]
    dr_value = s.dr_calc()[0]
    noise_value = s.noise_calc()

    freq.append(freq_value)
    dr.append(dr_value)
    noise.append(abs(noise_value))

        

    end_time = time.time()

    duration = end_time - start_time
    print('duration=',duration) 

    # Summary stores an array of the first-harmonic amplitude
##############################################################################################
    #LIVE PLOT FIGURE
    plt.figure()
############################################################################################
#SIGNAL SPLITTING

##    data = loadtxt('firstharmonic.txt')
####    
##    # Split data into time and amplitude
##    t = data[:, 0]
##    A = data[:, 1]
##    Amp = A[0]; dr = s.dr_calc()[0]; freq = s.freq_calc()[0]; w = 2*3.14159 * freq
##    signal = (Amp * (exp(-(dr*t))) * cos(w*t))
######
##    freq_err = s.freq_calc()[1]
##    dr_err = s.dr_calc()[1]
####    
##    error = sqrt(dr_err**2 + freq_err**2)
##    error_tot = 100*error
####
##    plt.plot(t, A, '-k', linewidth=2.0, label='data')    
##    plt.plot(t, signal, '-o', markersize=0.1, label = u'signal, w={:.3f}, dr={:.3f}, Δ={:.2f}% '.format(w,dr,error_tot), linewidth=1.0)
##    plt.fill_between(t, signal*(1+error), signal*(1-error))
########################################################################################################################################    
##ANALYTICAL SOLUTIONS
##    analytical1 = A[0] * exp(-0.153 * t) * cos(1.416 * t)
##    analytical2 = A[0] * exp(-0.168 * t) * cos(1.33 * t)
##    analytical2_error = 0.002/0.168 + 0.16/1.33

##    plt.plot(t, analytical2, markersize=0.1, label = u'w=1.33, dr=0.168, Δ=13.22%' , linewidth=1.0)
##    plt.fill_between(t, analytical2*(1+analytical2_error), analytical2*(1-analytical2_error))
##
##    plt.plot(t, analytical1, markersize=0.1, label = 'w=1.416, dr=0.153', linewidth=1.0)
##########################################################################################################################################
## FITTING FUNCTIONS
##    def log_fit(ncells_values, *params):
##        a = params[0]
##        b = params[1]
##        x0 = params[2]
##
##        return a + b * log10(abs(ncells_values - x0))
##
##    guess = [0.5,0.04,39.5]
##    popt, pcov = curve_fit(log_fit, ncells_values, freq, p0=guess)
##    yfit = log_fit(ncells_values, *popt)
##    log_R_squared =sqrt(mean(freq - yfit)**2)
##
##    plt.plot(ncells_values, yfit, '-', label=f'''Log Fit: y = {popt[0]:.2f} + {popt[1]:.2f} + log(x - {popt[2]:.2f});
##                                            square residual = {log_R_squared}''')
##
##    
##    def linear_fit(ncells_values, *params):
##        c = params[0]
##        m = params[1]
##        x0 = params[2]
##
##        return m * (ncells_values - x0) + c
##
##    guess1 = [0.5,0.04,39.5]
##    popt_1, pcov_1 = curve_fit(linear_fit, ncells_values, freq, p0=guess1)
##    yfit1 = linear_fit(ncells_values, *popt_1)
##    lin_R_squared =sqrt(mean(freq - yfit1)**2)
##
##    plt.plot(ncells_values, yfit1, '-', label=f'''Linear Fit: y = {popt_1[0]:.2f} * (x - {popt_1[1]:.2f}) + {popt_1[2]:.2f};
##             square residual = {lin_R_squared}''')
############################################################################################################################################
##PHYSICAL DEPENDENCE OF SYSTEM SIZES
##    plt.plot(L_values, noise, '-o', label='noise')
##    plt.plot(L_values, dr, '-o', label='damping rate')
##    plt.plot(L_values, freq, '-o', label='freq')
##########################################################################################################
##PLOTTING
    plt.plot(s.t, s.firstharmonic, '-k', label='first harmonic')
    plt.yscale('log')
    plt.xlabel("time")
    plt.ylabel("Amplitude [Normalised]")
    plt.legend()
    plt.ioff() # This so that the windows stay open
    plt.show()
##    
