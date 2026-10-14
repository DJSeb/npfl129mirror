#!/usr/bin/env python3

# Me: 4ae50bd7-2516-11ec-986f-f39926f24a9c (David jaromír Šebánek)
# Team member 1: a652f90b-2220-11ec-986f-f39926f24a9c (Norbert Horváth)
# Team member 2: 5212d2e4-2516-11ec-986f-f39926f24a9c (Samuel Serafín)

import argparse

import numpy as np
import sklearn.datasets

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate binary classification data with labels {-1, 1}.
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, flip_y=0, class_sep=2, random_state=args.seed)
    target = 2 * target - 1

    # DONE: Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    bias_arr = np.ones(data.shape[0])
    data_w_bias = np.column_stack((data, bias_arr))

    # Generate initial perceptron weights.
    weights = np.zeros(data_w_bias.shape[1])

    done = False
    while not done:
        done = True
        permutation = generator.permutation(data.shape[0])

        # DONE: Implement the perceptron algorithm, notably one iteration
        # over the training data in the order of `permutation`. During the
        # training data iteration, perform the required updates to the `weights`
        # for incorrectly classified examples. If all training instances are
        # correctly classified, set `done=True`, otherwise set `done=False`.
        for i in permutation:
            example_vctr = data_w_bias[i]
            computed = np.matmul(example_vctr, weights)
            trgt_comp = target[i] * computed

            if trgt_comp <= 0:
                weights += target[i] * data_w_bias[i]
                done = False

        if args.plot and not done:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4*3, 4.8*3))
                plt.subplot(3, 3, 1 + len(plt.gcf().get_axes()))
            plt.scatter(data[:, 0], data[:, 1], c=target)
            xs = np.linspace(*plt.gca().get_xbound() + (50,))
            ys = np.linspace(*plt.gca().get_ybound() + (50,))
            plt.contour(xs, ys, [[[x, y, 1] @ weights for x in xs] for y in ys], levels=[0])
            plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights = main(main_args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))
