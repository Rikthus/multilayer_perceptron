import sys
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def load_dataset() -> pd.DataFrame:
    assert len(sys.argv) == 2, "usage: select_features.py <dataset/path>"

    path = sys.argv[1]
    assert os.path.isfile(path), "invalid dataset path"

    col_names = ["ID", "State"] + [f"ft{nb}" for nb in range(30)]
    df = pd.read_csv(path, names=col_names)
    return df


def display_pair_plot(df: pd.DataFrame):
    df.drop(columns=["ID"], inplace=True)
    plot = sns.pairplot(df, hue="State")
    plot.figure.set_size_inches(24, 16)
    plt.show()


def main():
    try:
        df = load_dataset()
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        print(df)
        print(df.describe())
        display_pair_plot(df)
    except Exception as e:
        print(f"{e.__class__.__name__}: {e.args[0]}")


if __name__ == "__main__":
    main()
