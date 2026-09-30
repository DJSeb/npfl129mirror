#!/usr/bin/env python3

# Me: 4ae50bd7-2516-11ec-986f-f39926f24a9c (David jaromír Šebánek)
# Team member 1: a652f90b-2220-11ec-986f-f39926f24a9c (Norbert Horváth)
# Team member 2: 5212d2e4-2516-11ec-986f-f39926f24a9c (Samuel Serafín)

import argparse

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_size", default=40, type=int, help="Data size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--range", default=3, type=int, help="Feature order range")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> list[float]:
    # Create artificial noisy data.
    xs = np.linspace(0, 7, num=args.data_size)
    ys = np.sin(xs) + np.random.RandomState(args.seed).normal(0, 0.2, size=args.data_size)

    rmses = []

    features = np.zeros([args.data_size, args.range])
    for order in range(1, args.range + 1):
        # DONE: Create features `(x^1, x^2, ..., x^order)`, preferably in this ordering.
        # Note that you can just append `x^order` to the features from the previous iteration.
        for row in range(0, args.data_size):
            features[row][order - 1] = xs[row]**order

        # DONE: Split the data into a train set and a test set.
        # Use `sklearn.model_selection.train_test_split` method call, passing
        # arguments `test_size=args.test_size, random_state=args.seed`.
        datatrain, datatest, targettrain, targettest = sklearn.model_selection.train_test_split(
            features, ys, test_size=args.test_size, random_state=args.seed
        )

        # DONE: Fit a linear regression model `sklearn.linear_model.LinearRegression(tol=1e-15)`
        # on the train set using the `fit` method. We use a stricter tolerance `tol=1e-15`
        # as the default tolerance is not sufficient when using features of higher order.
        model = sklearn.linear_model.LinearRegression(tol=1e-15)
        fitted = model.fit(datatrain, targettrain)

        # DONE: Predict targets on the test set using the `predict` method of the trained model.
        predicted = fitted.predict(datatest)

        # DONE: Compute root mean square error on the test set predictions.
        # You can either do it manually, or you can look at the metrics offered
        # by the `sklearn.metrics` module.
        rmse = sklearn.metrics.root_mean_squared_error(targettest, predicted)

        rmses.append(rmse)

        if args.plot:
            # The plotting code assumes the train/test data/targets are in numpy arrays
            # `train_data`, `train_target`, `test_data`, `test_target`.
            import matplotlib.pyplot as plt
            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4*3, 4.8*3))
                plt.subplot(3, 3, 1 + len(plt.gcf().get_axes()))
            plt.plot(train_data[:, 0], train_target, "go")
            plt.plot(test_data[:, 0], test_target, "ro")
            plt.plot(np.linspace(xs[0], xs[-1], num=100),
                     model.predict(np.power.outer(np.linspace(xs[0], xs[-1], num=100), np.arange(1, order + 1))), "b")
            plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return rmses


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    rmses = main(main_args)
    for order, rmse in enumerate(rmses):
        print("Maximum feature order {}: {:.2f} RMSE".format(order + 1, rmse))
