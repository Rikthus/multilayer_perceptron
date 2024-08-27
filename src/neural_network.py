import numpy as np

from layer import Layer

OUTPUT_NEURONS = 1

from scipy.special import softmax
np.set_printoptions(precision=5)

class NeuralNetwork:
    def __init__(
        self,
        learning_rate: float,
        epochs: int,
        layers_dimensions: list[int],
        nb_features: int,
        batch_size: int,
    ) -> None:
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__layers = self.__init_layers(layers_dimensions, nb_features, learning_rate)

    def __init_layers(
        self, layers_dimensions: list[int], nb_features: int, learning_rate: float
    ) -> list[Layer]:
        """Initialize layers 

        Args:
            layers_dimensions (list[int]): _description_
            nb_features (int): _description_
            learning_rate (float): _description_

        Returns:
            list[Layer]: _description_
        """
        layers = []
        layers_dimensions.insert(0, nb_features)
        layers_dimensions.append(OUTPUT_NEURONS)

        for i in range(1, len(layers_dimensions)):
            if i == len(layers_dimensions) - 1:
                layers.append(Layer(layers_dimensions[i], layers_dimensions[i - 1], learning_rate, self.__softmax))
            else:
                layers.append(Layer(layers_dimensions[i], layers_dimensions[i - 1], learning_rate, self.__sigmoid))
        return layers
    
    def __sigmoid(self, Z: np.ndarray) -> np.ndarray:
        act = 1 / (1 + np.exp(-Z))
        return act
    
    def __softmax(self, Z: np.ndarray) -> np.ndarray:
        # exp_z = np.exp(Z)
        # sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
        # act = exp_z / sum_exp_z
        # act = softmax(Z, axis=1)
        act = 1 / (1 + np.exp(-Z))
        return act
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.__forward_pass(X)
        return self.__layers[-1].activations >= 0.5

    def __forward_pass(self, x_train: np.ndarray):
        self.__layers[0].forward_pass(x_train)
        for i in range(1, len(self.__layers)):
            self.__layers[i].forward_pass(self.__layers[i - 1].activations)

    def __back_propagation(self, x_train: np.ndarray, y_train: np.ndarray):
        m = y_train.shape[1]
        dZ = self.__layers[-1].activations - y_train
        
        for i in reversed(range(0, len(self.__layers))):
            if i == 0:
                prev_A = x_train
            else:
                prev_A = self.__layers[i - 1].activations
            self.__layers[i].back_propagation(dZ, m, prev_A)
            if i > 0:
                dZ = self.__layers[i].weights.T.dot(dZ) * prev_A * (1 - prev_A)
    
    def __update_gradients(self):
        for layer in self.__layers:
            layer.update()

    def fit_model(self, x_train: np.ndarray, y_train: np.ndarray):
        nb_samples = len(x_train)
        for _ in range(self.__epochs):
            batch_start_idx = 0
            while batch_start_idx < nb_samples:
                if batch_start_idx + self.__batch_size > nb_samples:
                    x_batch = x_train[batch_start_idx : nb_samples]
                    y_batch = y_train[batch_start_idx : nb_samples]
                else:
                    x_batch = x_train[batch_start_idx : batch_start_idx + self.__batch_size]
                    y_batch = y_train[batch_start_idx : batch_start_idx + self.__batch_size]
                self.__forward_pass(x_batch.T)
                self.__back_propagation(x_batch.T, y_batch.T)
                self.__update_gradients()
                batch_start_idx += self.__batch_size
                
    # def predict(self, x_test: np.ndarray) -> np.ndarray:
        

    def print_shape(self):
        for layer in self.__layers:
            layer.print_shape()

    def print_activations(self):
        for layer in self.__layers:
            layer.print_shape()
            print(layer.activations)
            
    def print_weights(self):
        for layer in self.__layers:
            layer.print_shape()
            print(layer.weights)
