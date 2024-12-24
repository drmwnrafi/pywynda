import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm  # Import tqdm for progress bar
from src.wynda import WyNDA
from src.gen_func import GenerateWideArray

# Lorenz system function
def lorentz_sys(sigma: float, rho: float, beta: float, state: np.array):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

# Parameters
init_state = np.array([1.0, 1.0, 1.0])
state = init_state
t_sim = [0.0, 25.0]
dt = 0.001
t = np.arange(t_sim[0], t_sim[1], dt)

sigma = 10.0
rho = 28.0
beta = 3.0

wynda = WyNDA(n_state=3, n_params=30, init_state=init_state, lambda_state=0.995, lambda_params=0.999)
widearray = GenerateWideArray(n_state=3)

def basis_function(input: np.array):
    basis = np.array([
        1,
        input[0], input[1], input[2],              
        input[0]**2, input[1]**2, input[2]**2,
        input[0]*input[1],                        
        input[0]*input[2],                        
        input[1]*input[2]                         
    ])
    return basis

# Simulation history
state_history = np.zeros((len(t), 3))
wynda_history = np.zeros((len(t), 3))
params_history = np.zeros((len(t), 30))

# Run the simulation with progress bar
for i, time in enumerate(tqdm(t, desc="Simulating Lorenz Attractor")):
    state += lorentz_sys(sigma, rho, beta, state) * dt
    basis = basis_function(state)
    Phi = widearray.custom(basis_function=basis)
    wynda_state, wynda_params = wynda.run(input=state, wide_array=Phi, dt=dt)
    wynda_history[i, :] = wynda_state
    state_history[i, :] = state
    params_history[i, :] = wynda_params

# Sample data for faster animation
sample_interval = 5  # Increase sampling interval to reduce the number of frames
state_history = state_history[::sample_interval]
wynda_history = wynda_history[::sample_interval]
t = t[::sample_interval]

# Create animation
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim((-20, 20))
ax.set_ylim((-30, 30))
ax.set_zlim((0, 50))

# Font sizes for labels and title
label_fontsize = 18
legend_fontsize = 18

ax.set_xlabel("X Axis", fontsize=label_fontsize)
ax.set_ylabel("Y Axis", fontsize=label_fontsize)
ax.set_zlabel("Z Axis", fontsize=label_fontsize)

# Line properties
line1, = ax.plot([], [], [], lw=1.5, color='r', label="True State")  # Increased linewidth
line2, = ax.plot([], [], [], lw=1.5, color='b', linestyle='--', label="WyNDA State")  # Increased linewidth

# Enhanced legend
ax.legend(fontsize=legend_fontsize)

def update(frame):
    line1.set_data(state_history[:frame, 0], state_history[:frame, 1])
    line1.set_3d_properties(state_history[:frame, 2])
    line2.set_data(wynda_history[:frame, 0], wynda_history[:frame, 1])
    line2.set_3d_properties(wynda_history[:frame, 2])
    return line1, line2

ani = FuncAnimation(fig, update, frames=len(t), interval=30, blit=True)

# Save as GIF
output_path = "lorenz_attractor.gif"
ani.save(output_path, writer=PillowWriter(fps=20))
print(f"Enhanced simulation saved as {output_path}")
