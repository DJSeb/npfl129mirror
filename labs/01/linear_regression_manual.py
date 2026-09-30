#!/usr/bin/env python3

# Me: 4ae50bd7-2516-11ec-986f-f39926f24a9c (David jaromír Šebánek)
# Team member 1: a652f90b-2220-11ec-986f-f39926f24a9c (Norbert Horváth)
# Team member 2: 5212d2e4-2516-11ec-986f-f39926f24a9c (Samuel Serafín)

import argparse

import numpy as np
import sklearn.datasets
from sklearn.utils import Bunch
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> float:
    # Load the diabetes dataset.
    dataset: Bunch = sklearn.datasets.load_diabetes()

    # The input data are in `dataset.data`, targets are in `dataset.target`.

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    # DONE: Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    feature_mtrx = dataset.data
    targets = dataset.target

    fmtrx_xsize = feature_mtrx.shape[0]
    bias_arr = np.ones(fmtrx_xsize)
    feature_mtrx_wbias = np.column_stack((feature_mtrx, bias_arr))

    # DONE: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    datatrain, datatest, targettrain, targettest = sklearn.model_selection.train_test_split(
        feature_mtrx_wbias, targets, test_size=args.test_size, random_state=args.seed
    )

    # DONE: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    transdtrain = np.transpose(datatrain)
    inversed_mult = np.linalg.inv(np.linalg.matmul(transdtrain, datatrain))
    leftwpart = np.linalg.matmul(inversed_mult, transdtrain)
    weights = np.dot(leftwpart, targettrain)

    # DONE: Predict target values on the test set.
    predicted = np.linalg.matmul(datatest, weights)

    # DONE: Manually compute root mean square error on the test set predictions.
    rmse = np.sqrt(np.mean((predicted - targettest)**2))

    return rmse


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(main_args)
    print("{:.2f}".format(rmse))
