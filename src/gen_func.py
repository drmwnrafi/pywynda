import numpy as np

class GenerateWideArray:
    def __init__(self, n_state:int):
        self.n_state = n_state

    def custom(self, basis_function:np.array):
        n_params = self.n_state*len(basis_function)
        full_matrix = np.zeros((self.n_state, n_params))
        for i in range(self.n_state):
            start_index = i * basis_function.shape[0]
            end_index = start_index + basis_function.shape[0]

            if end_index <= n_params:
                full_matrix[i, start_index:end_index] = basis_function
        return full_matrix
    
    def polynomial(self, order:int):
        # TODO
        pass

    def fourier(self,):
        # TODO
        pass

    def trigonometric(self,):
        # TODO
        pass