"""
Implementation of Discovering State-Space Representation of
Dynamical Systems From Noisy Data by Agus Hasan

PDF Link : https://www.researchgate.net/publication/382909160_Discovering_State-Space_Representation_of_Dynamical_Systems_from_Noisy_Data
"""
import numpy as np

class WyNDA:
  def __init__(self,
                 n_state:int,  # n
                 n_params:int,  # r
                 init_state:np.array=None,
                 lambda_state:float=0.65,
                 lambda_params:float=0.995,
                 R_state:np.array=None,
                 R_params:np.array=None,
                 P_state:np.array=None,
                 P_params:np.array=None):
    self.R_state = R_state if R_state is not None else 1 * np.eye(n_state, n_state)
    self.R_params = R_params if R_params is not None else 1 * np.eye(n_state, n_state)
    self.P_state = P_state if P_state is not None else 0.1 * np.eye(n_state, n_state)
    self.P_params = P_params if P_params is not None else 0.1 * np.eye(n_params, n_params)
    self.Gamma = np.eye(n_state, n_params)
    self.lambda_state = lambda_state
    self.lambda_params = lambda_params
    self.K_state = np.zeros((n_state, n_state))
    self.K_params = np.zeros((n_params, n_state))
    self.state = np.zeros(n_state) if init_state is None else init_state
    self.params = np.zeros(n_params)
    self.n_state = n_state
    self.n_params = n_params

  def update_gain(self,):
    self.K_state = self.P_state @ np.linalg.inv(self.P_state + self.R_state)
    self.K_params = self.P_params @ self.Gamma.T @ \
              np.linalg.inv(self.Gamma @ self.P_params @ self.Gamma.T + self.R_params)
    self.Gamma = (np.eye(self.n_state) - self.K_state) @ self.Gamma

  def update_model(self, base, phi, dt):
    noise = np.random.normal(scale=0.01, size=self.state.shape)
    self.state = self.state + base * dt + phi @ self.params
    self.params = self.params
    self.P_state = 1/self.lambda_state * (np.eye(self.n_state) - self.K_state) @ self.P_state
    self.P_params = 1/self.lambda_params * (np.eye(self.n_params) - self.K_params @ self.Gamma) @ self.P_params
    self.Gamma = self.Gamma - phi

  def estimate(self, input):
    self.state = self.state + (self.K_state + self.Gamma @ self.K_params) @ (input - self.state)
    self.params = self.params - self.K_params @ (input - self.state)

  def run(self, input:np.array, wide_array:np.array, dt:float, base:np.array=None):
    if base is None :
      base = np.zeros(self.n_state)
    self.update_gain()
    self.estimate(input)
    self.update_model(base, wide_array, dt)
    return self.state, self.params