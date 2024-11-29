# Importing necessary libraries
import scipy as sci
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Non-Dimensionalisation
G = 1  # N-m^2/kg^2

# Reference quantities
m_nd = 1  # kg
r_nd = 1  # m
v_nd = 1  # m/s
t_nd = 1  # s

# Net constants
K1 = G * t_nd * m_nd / (r_nd**2 * v_nd)
K2 = v_nd * t_nd / r_nd

# Define masses
m1, m2, m3 = 1, 1, 1  # Masses of stars

# Define initial position vectors in 2D (x, y)
r1 = [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]  # m
r2 = [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]  # m
r3 = [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]  # m

# Convert position vectors to arrays
r1 = np.array(r1)
r2 = np.array(r2)
r3 = np.array(r3)

# Define initial velocities in 2D (vx, vy)
v1 = [np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1)]  # m/s
v2 = [np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1)]  # m/s
v3 = [np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1)]  # m/s

# Convert velocity vectors to arrays
v1 = np.array(v1)
v2 = np.array(v2)
v3 = np.array(v3)

# Define the 2D differential equations
def ThreeBodyEquations(t, w):
    # Unpack all the variables from the array "w"
    r1 = w[:2]
    r2 = w[2:4]
    r3 = w[4:6]
    v1 = w[6:8]
    v2 = w[8:10]
    v3 = w[10:12]

    # Find distances between the three bodies
    r12 = sci.linalg.norm(r2 - r1)
    r13 = sci.linalg.norm(r3 - r1)
    r23 = sci.linalg.norm(r3 - r2)

    # Define the derivatives according to the equations
    dv1bydt = K1 * m2 * (r2 - r1) / r12**3 + K1 * m3 * (r3 - r1) / r13**3
    dv2bydt = K1 * m1 * (r1 - r2) / r12**3 + K1 * m3 * (r3 - r2) / r23**3
    dv3bydt = K1 * m1 * (r1 - r3) / r13**3 + K1 * m2 * (r2 - r3) / r23**3
    dr1bydt = K2 * v1
    dr2bydt = K2 * v2
    dr3bydt = K2 * v3

    # Package the derivatives into one final size-12 array
    r_derivs = np.concatenate((dr1bydt, dr2bydt, dr3bydt))
    v_derivs = np.concatenate((dv1bydt, dv2bydt, dv3bydt))
    derivs = np.concatenate((r_derivs, v_derivs))
    return derivs

# Package initial parameters
init_params = np.array([r1, r2, r3, v1, v2, v3]).flatten()
time_span = (0, 200)  # Time span for 20 orbital years
time_eval = np.linspace(0, 200, 1000)  # Evaluation points

# Run the ODE solver using solve_ivp
three_body_sol = solve_ivp(
    ThreeBodyEquations, time_span, init_params, t_eval=time_eval, method="RK45"
)

# Store the position solutions into three distinct arrays
r1_sol = three_body_sol.y[:2, :].T
r2_sol = three_body_sol.y[2:4, :].T
r3_sol = three_body_sol.y[4:6, :].T

# Set up the figure for animation
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_xlabel("x-coordinate", fontsize=14)
ax.set_ylabel("y-coordinate", fontsize=14)
ax.set_title("Live Animation of 2D 3-Body Orbits", fontsize=14)

# Create objects for the animation
star1, = ax.plot([], [], "o", color="mediumblue", label="Star 1", markersize=8)
star2, = ax.plot([], [], "o", color="red", label="Star 2", markersize=8)
star3, = ax.plot([], [], "o", color="gold", label="Star 3", markersize=8)
trail1, = ax.plot([], [], "-", color="mediumblue", linewidth=1)
trail2, = ax.plot([], [], "-", color="red", linewidth=1)
trail3, = ax.plot([], [], "-", color="gold", linewidth=1)
ax.legend()

# Initialize the animation
def init():
    star1.set_data([], [])
    star2.set_data([], [])
    star3.set_data([], [])
    trail1.set_data([], [])
    trail2.set_data([], [])
    trail3.set_data([], [])
    return star1, star2, star3, trail1, trail2, trail3

# Update the animation at each frame
# Update the animation at each frame
def update(frame):
    # Safeguard frame indexing to avoid issues
    if frame < len(r1_sol):
        # Set positions for stars
        star1.set_data([r1_sol[frame, 0]], [r1_sol[frame, 1]])
        star2.set_data([r2_sol[frame, 0]], [r2_sol[frame, 1]])
        star3.set_data([r3_sol[frame, 0]], [r3_sol[frame, 1]])

        # Update the trails
        trail1.set_data(r1_sol[:frame + 1, 0], r1_sol[:frame + 1, 1])
        trail2.set_data(r2_sol[:frame + 1, 0], r2_sol[:frame + 1, 1])
        trail3.set_data(r3_sol[:frame + 1, 0], r3_sol[:frame + 1, 1])
    return star1, star2, star3, trail1, trail2, trail3


# Create the animation
ani = FuncAnimation(fig, update, frames=len(time_eval), init_func=init, blit=True, interval=20)

plt.show()
