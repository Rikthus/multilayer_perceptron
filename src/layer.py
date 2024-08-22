import numpy as np

class Layer:
    def __init__(self, nb_neurons: int, nb_input: int, learning_rate: float) -> None:
        self.__W = np.random.randn(nb_neurons, nb_input)
        self.__b = np.random.randn(nb_neurons, 1)
        self.__l_rate = learning_rate
        
    def forward_pass(self, activations: np.ndarray):
        W = self.__W
        A = activations
        b = self.__b
        
        Z = W.dot(A) + b
        self.__activations = 1 / (1 + np.exp(-Z))
        
    def back_propagation(self, dZ: np.ndarray, m: int, prev_activations: np.ndarray):
        self.__dW = (1 / m) * dZ.dot(prev_activations.T)
        self.__db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
    def update(self):
        self.__W = self.__W - self.__l_rate * self.__dW
        self.__b = self.__b - self.__l_rate * self.__db
        
    def print_shape(self):
        print(self.__W.shape)
        
    @property
    def activations(self) -> np.ndarray:
        return self.__activations
    
    @property
    def weights(self) -> np.ndarray:
        return self.__W
