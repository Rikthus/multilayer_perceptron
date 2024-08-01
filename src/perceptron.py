import numpy as np


THRESHOLD = 0.5


class   Perceptron:
    def __init__(self, learning_rate: float, nb_features):
        self.__learning_rate = learning_rate
        self.__init_weights_and_bias(nb_features)
        self.__cost = []

    def __init_weights_and_bias(self, nb_features):
        self.__W = np.random.rand(nb_features, 1)
        self.__b = np.random.randn(1)
        
    def __compute_cost(self, Y: np.ndarray):
        m = Y.shape[0]
        epsilon = 0.00001
        L = (-1 / m) * np.sum(Y * np.log(self.__A + epsilon) + (1 - Y) * np.log(1 - self.__A + epsilon))
        self.__cost.append(L) 
        
    def __update_gradients(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
        m = X.shape[0]
        dW = (1 / m) * X.T.dot(self.__A - Y)
        db = (1 / m) * np.sum(self.__A - Y)
        
        self.__W -= self.__learning_rate * dW
        self.__b -= self.__learning_rate * db
        
    def __model(self, X: np.ndarray) -> np.ndarray:
        Z = X.dot(self.__W) + self.__b
        A = 1 / (1 + np.exp(-Z))
        return A

    def train_model(self, X: np.ndarray, Y: np.ndarray):
        self.__A = self.__model(X)
        self.__compute_cost(Y)
        self.__update_gradients(X, Y)
        
    def predict(self, X: np.ndarray) -> str:
        A = self.__model(X)
        return A >= THRESHOLD

        
    @property
    def weights(self) -> np.ndarray:
        return self.__W
    
    
    @property
    def cost(self) -> float:
        return self.__cost

