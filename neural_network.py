class   NeuralNetwork:
    def __init__(self, learning_rate: float, epochs: int, nb_layers: int, nb_neurons: int) -> None:
        self.__l_rate = learning_rate
        self.__epochs = epochs
        self.__layers = build_layers(layers_neurons)