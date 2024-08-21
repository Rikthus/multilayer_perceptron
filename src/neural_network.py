import numpy as np

from layer import Layer


class NeuralNetwork:
    def __init__(
        self,
        learning_rate: float,
        epochs: int,
        layers_dimensions: list[int],
        nb_features: int,
    ) -> None:
        self.__l_rate = learning_rate
        self.__epochs = epochs
        self.__layers = self.__init_layers(layers_dimensions, nb_features)

    def __init_layers(
        self, layers_dimensions: list[int], nb_features: int
    ) -> list[Layer]:
        layers = []
        layers_dimensions.insert(0, nb_features)
        layers_dimensions.append(1)

        for i in range(1, len(layers_dimensions)):
            layers.append(Layer(layers_dimensions[i], layers_dimensions[i - 1]))
        return layers

    def __forward_pass(self, x_train: np.ndarray):
        self.__layers[0].forward_pass(x_train)
        for i in range(1, len(self.__layers)):
            self.__layers[i].forward_pass(self.__layers[i - 1].activations)

    def __back_propagation(self, x_train: np.ndarray, y_train: np.ndarray):
        pass

    def fit_model(self, x_train: np.ndarray, y_train: np.ndarray):
        for _ in range(self.__epochs):
            self.__forward_pass(x_train)
            self.__back_propagation(x_train, y_train)

    def print_shape(self):
        for layer in self.__layers:
            layer.print_shape()

    def print_activations(self):
        for layer in self.__layers:
            layer.print_shape()
            print(layer.activations)
