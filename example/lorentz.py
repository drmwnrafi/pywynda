import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.wynda import WyNDA
from src.gen_func import GenerateWideArray

def lorentz_sys(sigma:float, rho:float, beta:float, state:np.array):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

init_state = np.array([1.0, 1.0, 1.0])
state = init_state
t_sim = [0.0, 5.0]
dt = 0.001
t = np.arange(t_sim[0], t_sim[1], dt)

sigma = 10.0
rho = 28.0
beta = 3.0

wynda = WyNDA(3, 30, init_state, 0.995, 0.999)
widearray = GenerateWideArray(3)

def basis_function(input:np.array):
    basis = np.array([
        1,
        input[0], input[1], input[2],              
        input[0]**2, input[1]**2, input[2]**2,
        input[0]*input[1],                        
        input[0]*input[2],                        
        input[1]*input[2]                         
    ])
    return basis

state_history = np.zeros((len(t), 3))
wynda_history = np.zeros((len(t), 3))
params_history = np.zeros((len(t), 30))

for i, time in enumerate(t):
    state += lorentz_sys(sigma, rho, beta, state) * dt
    basis = basis_function(state)
    Phi = widearray.custom(basis)
    wynda_state, wynda_params = wynda.run(state, Phi, dt)
    wynda_history[i, :] = wynda_state
    state_history[i, :] = state
    params_history[i, :] = wynda_params

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(state_history[:, 0], state_history[:, 1], state_history[:, 2], lw=2, color='r', linestyle='-')
ax.plot(wynda_history[:, 0], wynda_history[:, 1], wynda_history[:, 2], lw=2, color='b', linestyle='--')
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")
plt.show()