import numpy as np

class Layer:
    def __init__(self, nb_neurons: int, nb_input: int) -> None:
        self.__W = np.random.randn(nb_neurons, nb_input)
        self.__b = np.random.randn(nb_neurons, 1)
        
    def forward_pass(self, activations: np.ndarray):
        W = self.__W
        A = activations
        b = self.__b
        
        Z = W.dot(A) + b
        self.__activations = 1 / (1 + np.exp(-Z))
        
    def print_shape(self):
        print(self.__W.shape)
        
        
    @property
    def activations(self) -> np.ndarray:
        return self.__activations
