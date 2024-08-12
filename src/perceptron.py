import numpy as np


ACTIVATION_THRESHOLD = 0.5


class   Perceptron:
    """Artificial neuron that takes a learning rate, and some data for training.
    """
    def __init__(self, learning_rate: float, X: np.ndarray, Y: np.ndarray):
        self.__learning_rate = learning_rate
        self.__X = X
        self.__Y = Y
        self.__init_weights_and_bias()
        self.__cost = []

    def __init_weights_and_bias(self):
        nb_features = len(self.__X[0])
        self.__W = np.random.rand(nb_features, 1)
        self.__b = np.random.randn(1)
        
    def __compute_cost(self):
        Y = self.__Y
        A = self.__A
        m = Y.shape[0]
        epsilon = 1e-15
        L = (-1 / m) * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
        self.__cost.append(L) 
        
    def __update_gradients(self) -> tuple[np.ndarray, float]:
        X = self.__X
        Y = self.__Y
        A = self.__A
        m = X.shape[0]
        
        dW = (1 / m) * X.T.dot(A - Y)
        db = (1 / m) * np.sum(A - Y)
        
        self.__W -= self.__learning_rate * dW
        self.__b -= self.__learning_rate * db
        
    def __model(self, X: np.ndarray) -> np.ndarray:
        Z = X.dot(self.__W) + self.__b
        A = 1 / (1 + np.exp(-Z))
        return A

    def train_model(self):
        self.__A = self.__model(self.__X)
        self.__compute_cost()
        self.__update_gradients()
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        A = self.__model(X_test)
        return A >= ACTIVATION_THRESHOLD


    @property
    def weights(self) -> np.ndarray:
        return self.__W
    
    
    @property
    def cost(self) -> float:
        return self.__cost

