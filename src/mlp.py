import argparse
import os

import pandas as pd
import sklearn.model_selection as skl_select
from sklearn.preprocessing import MinMaxScaler

# from sklearn.metrics import accuracy_score
# from sklearn.datasets import make_blobs
# import h5py
# import numpy as np
from neural_network import NeuralNetwork

# implement softmax activation for output_layer

# implement predict of x_test compared to y_test using the binary cross_entropy
# error function

# display train and test data set shapes
# display each epochs: epochs, precision metrics
# implement learning curves graphs after train

# save network topologie and weights in a file


# BONUSES

# multiple weights init strategies
# multiple activation functions
# implement early stopping
# add multiples metrics in learning phase

# historic of metrics ?
# a display of multiple learning curves on the same graph ?

# add thorough feature selection


def load_dataset(path: str) -> pd.DataFrame:
    """Load a csv dataset, name columns, drop NaN and duplicate rows

    Args:
        path (str): path to the dataset csv file to load

    Returns:
        pd.DataFrame: data from a csv file in a pandas dataframe
    """
    assert os.path.isfile(path), "invalid dataset path"

    col_names = ["ID", "Diagnosis"] + [f"ft{nb}" for nb in range(30)]
    df = pd.read_csv(path, names=col_names)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    pd.set_option("future.no_silent_downcasting", True)
    df.replace({"Diagnosis": {"M": 1, "B": 0}}, inplace=True)
    return df


def validate_layers(value):
    try:
        int_list = [int(x) for x in value.split(",")]

        if len(int_list) > 100:
            raise ValueError("List should contain at most 100 values")
        
        if len(int_list) < 3:
            raise ValueError("List should contain at least 3 values")

        if not all(0 < x < 100 for x in int_list):
            raise ValueError("All values should be positive and less than 100")

        if not all(x > 0 for x in int_list):
            raise ValueError("All values should be greater than 0")

        return int_list
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def parse_arguments() -> list[any]:
    """Parse arguments given to the program and return the list of arguments

    Returns:
        list[any]: list of arguments provided by the user or default values
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset")
    parser.add_argument(
        "--layers",
        type=validate_layers,
        default=[1, 1, 1],
        help='Layers dimensions in this format "<neurons in layer 1>, <neurons in layer 2>, etc...". For example to create 3 layers with respectively 20 and 5 neurons: "20, 5".',
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate [0.00001 - 1.0]."
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of iterations [1 - 10000]."
    )
    parser.add_argument(
        "--batch_size", type=int, default=0, help="Choose batch size for each gradient computing [1 - max_samples]."
    )

    args = parser.parse_args()

    args.lr = min(1.0, max(args.lr, 0.001))
    args.epochs = min(1000, max(args.epochs, 1))

    return args

def adapt_batch_size(arg_size: int, dataset_size: int) -> int: 
   if arg_size == 0 or arg_size >= dataset_size:
       return dataset_size
   else:
       return arg_size


def main():
    args = parse_arguments()
    df = load_dataset(args.dataset)
    
    scaler = MinMaxScaler()
    x_df = df.drop(columns=["ID", "Diagnosis"])
    scaled_x_df = scaler.fit_transform(x_df)
    y_df = df["Diagnosis"]
    x_train, x_test, y_train, y_test = skl_select.train_test_split(
        scaled_x_df, y_df, test_size=0.2, random_state=42
    )
    y_train = y_train.to_numpy().reshape((y_train.shape[0], 1)).astype(float)
    y_test = y_test.to_numpy().reshape((y_test.shape[0], 1)).astype(float)

    args.batch_size = adapt_batch_size(args.batch_size, len(x_train))
    network = NeuralNetwork(args.lr, args.epochs, args.layers, x_train.shape[1], args.batch_size)
    network.fit_model(x_train, y_train)
    network.save_topologie()
    print(network.predict(x_test.T) == y_test.T)
    # network.print_shape()
    # network.print_activations()
    # network.print_weights()


if __name__ == "__main__":
    main()


# def load_data():
#     train_dataset = h5py.File('datasets/trainset.hdf5', "r")
#     X_train = np.array(train_dataset["X_train"][:]) # your train set features
#     y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

#     test_dataset = h5py.File('datasets/testset.hdf5', "r")
#     X_test = np.array(test_dataset["X_test"][:]) # your train set features
#     y_test = np.array(test_dataset["Y_test"][:]) # your train set labels

#     return X_train, y_train, X_test, y_test

# def flatten_images(data):
#     lst = []
#     for elem in data:
#         lst.append(elem.flatten())

#     return np.array(lst)

# def fit_model(x: pd.DataFrame, y: pd.DataFrame, lr: float, epochs: int, x_test, y_test):
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(x)
#     Y = y
#     p = Perceptron(lr, X, Y)
#     for _ in range(epochs):
#         p.train_model()
#     plt.plot(p.cost)
#     plt.show()

#     x_test = scaler.fit_transform(x_test)
#     pred = p.predict(x_test)

#     acc = accuracy_score(y_test, pred)
#     print(acc)

# def main():
#     args = parse_arguments()
#     X_train, y_train, X_test, y_test = load_data()
#     X_train = flatten_images(X_train)
#     X_test = flatten_images(X_test)

#     fit_model(X_train, y_train, args.lr, args.epochs, X_test, y_test)
