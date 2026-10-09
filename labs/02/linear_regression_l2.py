#!/usr/bin/env python3

# Me: 4ae50bd7-2516-11ec-986f-f39926f24a9c (David jaromír Šebánek)
# Team member 1: a652f90b-2220-11ec-986f-f39926f24a9c (Norbert Horváth)
# Team member 2: 5212d2e4-2516-11ec-986f-f39926f24a9c (Samuel Serafín)

import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=13, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Load the diabetes dataset.
    dataset = sklearn.datasets.load_diabetes()

    # DONE: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    datatrain, datatest, trgttrain, trgttest = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed
    )

    lambdas = np.geomspace(0.01, 10, num=500)
    # DONE: Using `sklearn.linear_model.Ridge`, fit the train set using
    # L2 regularization, employing the above defined lambdas.
    # For every model, compute the root mean squared error and return the
    # lambda producing lowest RMSE and the corresponding RMSE.
    model = sklearn.linear_model.Ridge(lambdas[0]).fit(datatrain, trgttrain)
    
    predicted = model.predict(datatest)

    best_rmse = sklearn.metrics.root_mean_squared_error(trgttest, predicted)
    best_lambda = lambdas[0]

    rmses = [best_rmse]

    for lamb in lambdas[1:]:
        model = sklearn.linear_model.Ridge(lamb).fit(datatrain, trgttrain)
        predicted = model.predict(datatest)
        rmse = sklearn.metrics.root_mean_squared_error(trgttest, predicted)

        rmses.append(rmse)

        if rmse < best_rmse:
            best_lambda = lamb
            best_rmse = rmse

    if args.plot:
        # This block is not required to pass in ReCodEx; however, it is useful
        # to learn to visualize the results. If you collect the respective
        # results for `lambdas` to an array called `rmses`, the following lines
        # will plot the result if you add `--plot` argument.
        import matplotlib.pyplot as plt
        plt.plot(lambdas, rmses)
        plt.xscale("log")
        plt.xlabel("L2 regularization strength $\\lambda$")
        plt.ylabel("RMSE")
        if plt.isinteractive():
            plt.show()
        else:
            import pathlib
            import os
            thisfname = pathlib.Path(__file__).resolve()

            plotsdir = thisfname.parent / 'plots'
            os.mkdir(plotsdir) if not pathlib.Path.exists(plotsdir) else None
            
            testsize_as_str = f'_test_size_{args.test_size}_'.replace('.', 'd')
            imagename = thisfname.name[:-3] + testsize_as_str + '.jpeg'
            
            plotpath = plotsdir / imagename 
            plt.savefig(plotpath, transparent=True, bbox_inches="tight")

    return best_lambda, best_rmse


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    best_lambda, best_rmse = main(main_args)
    print("{:.2f} {:.2f}".format(best_lambda, best_rmse))
