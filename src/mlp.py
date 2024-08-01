import argparse
import os
import pandas as pd
import sklearn.model_selection as skl_select
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from perceptron import Perceptron


# 1 perceptron
# multicouches
# graphs
# algo optimize
# select features

def load_dataset(path: str) -> pd.DataFrame:
    """Load a csv dataset, name columns, drop NaN and duplicate rows

    Returns:
        pd.DataFrame: data from a csv file in a pandas dataframe
    """
    assert os.path.isfile(path), "invalid dataset path"

    col_names = ["ID", "Diagnosis"] + [f"ft{nb}" for nb in range(30)]
    df = pd.read_csv(path, names=col_names)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    pd.set_option('future.no_silent_downcasting', True)
    df.replace({"Diagnosis": {"M": 1, "B": 0}}, inplace=True)
    return df


def parse_arguments() -> list[any]:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("dataset")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate [0.0001 - 1.0]")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of iterations [1 - 10000]")
    
    args = parser.parse_args()
    
    args.lr = min(1.0, max(args.lr, 0.0001))
    args.epochs = min(10000, max(args.epochs, 1))
    
    return args


def fit_model(x: pd.DataFrame, y: pd.DataFrame, lr: float, epochs: int):
    X = x.to_numpy()
    Y = y.to_numpy().reshape(y.shape[0], 1).astype(float)
    p = Perceptron(lr, len(x.columns))
    for _ in range(epochs):
        p.train_model(X, Y)
    plt.plot(p.cost)
    plt.show()


def main():
    args = parse_arguments()
    df = load_dataset(args.dataset)
    
    x_df = df.drop(columns=["ID", "Diagnosis"])
    y_df = df["Diagnosis"]
    x_train, x_test, y_train, y_test = skl_select.train_test_split(x_df, y_df, test_size=0.2, random_state=42)
    
    fit_model(x_train, y_train, args.lr, args.epochs)


if __name__ == "__main__":
    main()