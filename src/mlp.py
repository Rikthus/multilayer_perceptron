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

        if not all(0 < x < 100 for x in int_list):
            raise ValueError("All values should be positive and less than 100")

        if not any(x > 0 for x in int_list):
            raise ValueError("At least one value should be greater than zero")

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
        default=1,
        help='Layers dimensions in this format "<neurons in layer 1>, <neurons in layer 2>, etc...". For example to create 2 layers with respectively 20 and 5 neurons: "20, 5"',
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate [0.00001 - 1.0]"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of iterations [1 - 10000]"
    )

    args = parser.parse_args()

    args.lr = min(1.0, max(args.lr, 0.001))
    args.epochs = min(1000, max(args.epochs, 1))

    return args


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

    network = NeuralNetwork(args.lr, args.epochs, args.layers, x_train.shape[1])
    network.fit_model(x_train.T, y_train.T)
    # network.print_shape()
    # network.print_activations()


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
