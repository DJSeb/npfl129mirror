#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization strength")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=92, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

def _construct_args_string(args: argparse.Namespace, wanted_labels: list[str]) -> str:
    all_args = args._get_kwargs()
    cleared_args = { argname: argval for argname, argval in all_args if argname in wanted_labels }
    outstr = ""
    for argname, argval in cleared_args.items():
        argstr = f'_{argname}_{argval}'
        if isinstance(argval, float):
            argstr = argstr.replace('.', 'd')
        outstr += argstr
    
    return outstr

def _save_plot_image(args: argparse.Namespace, plt) -> None:
    import matplotlib.pyplot as plt
    import pathlib
    import os
    thisfname = pathlib.Path(__file__).resolve()

    plotsdir = thisfname.parent / 'plots'
    os.mkdir(plotsdir) if not pathlib.Path.exists(plotsdir) else None
    
    args_infix = _construct_args_string(
        args,
        ['batch_size', 'data_size', 'epochs', 'l2', 'learning_rate', 'test_size']
    )

    imagename = thisfname.name[:-3] + args_infix + '.jpeg'
    
    plotpath = plotsdir / imagename
    plt.savefig(plotpath, transparent=True, bbox_inches="tight")
    #print(f'Saved as "{imagename}"')

def main(args: argparse.Namespace) -> tuple[list[float], float, float]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset.
    data, target = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)

    # DONE: Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    bias_arr = np.ones(data.shape[0])
    data_w_bias = np.column_stack((data, bias_arr))

    # DONE: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data_w_bias, target, test_size=args.test_size, random_state=args.seed
    )

    # Generate initial linear regression weights.
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # DONE: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `len(train_data)`.
        #
        # The gradient for the input example $(x_i, t_i)$ is
        # - $(x_i^T weights - t_i) x_i$ for the unregularized loss (1/2 MSE loss),
        # - $args.l2 * weights_with_bias_set_to_zero$ for the L2 regularization loss,
        #   where we set the bias to zero because the bias should not be regularized,
        # and the SGD update is
        #   weights = weights - args.learning_rate * gradient

        for chunk_start in range(0, len(permutation), args.batch_size):
            batch_indices = permutation[chunk_start:chunk_start + args.batch_size]
            databatch = train_data[batch_indices]
            targetbatch = train_target[batch_indices]
            # Matrix multiplication is also doable via @ operator
            unregularized = np.matmul(
                np.matmul(databatch, weights) - targetbatch,
                databatch
            ) / args.batch_size

            weights_w_zbias = weights.copy()
            weights_w_zbias[-1] = 0
            reg_loss = args.l2 * weights_w_zbias

            gradient = unregularized + reg_loss
            weights = weights - args.learning_rate * gradient

        train_predicted = np.matmul(train_data, weights)
        test_predicted = np.matmul(test_data, weights)

        # DONE: Append current RMSE on train/test to `train_rmses`/`test_rmses`.
        train_rmses.append(sklearn.metrics.root_mean_squared_error(train_target, train_predicted))
        test_rmses.append(sklearn.metrics.root_mean_squared_error(test_target, test_predicted))

    # DONE: Compute into `explicit_rmse` test data RMSE when fitting
    # `sklearn.linear_model.LinearRegression` on `train_data` (ignoring `args.l2`).
    fmodel = sklearn.linear_model.LinearRegression().fit(train_data, train_target)
    explicit_rmse = sklearn.metrics.root_mean_squared_error(test_target, fmodel.predict(test_data))

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.legend()
        if plt.isinteractive():
            plt.show()
        else:
            _save_plot_image(args, plt)

    return weights, test_rmses[-1], explicit_rmse


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, sgd_rmse, explicit_rmse = main(main_args)
    print("Test RMSE: SGD {:.3f}, explicit {:.1f}".format(sgd_rmse, explicit_rmse))
    print("Learned weights:", *("{:.3f}".format(weight) for weight in weights[:12]), "...")
