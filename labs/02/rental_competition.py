#!/usr/bin/env python3

# Me: 4ae50bd7-2516-11ec-986f-f39926f24a9c (David jaromír Šebánek)
# Team member 1: a652f90b-2220-11ec-986f-f39926f24a9c (Norbert Horváth)
# Team member 2: 5212d2e4-2516-11ec-986f-f39926f24a9c (Samuel Serafín)

import argparse
import lzma
from dataclasses import dataclass
import os
from pathlib import Path 
import pickle
import sys
from textwrap import dedent
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import sklearn

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path",
    default=str((Path(__file__).resolve().parent / "rental_competition.model")),
    type=str,
    help="Model path"
)

@dataclass
class SgdHyperparameters:
    epochs = 800
    batch_size = 64
    learning_rate = 0.01
    l2 = 0.0001

def print_epoch_metrics_outside_recodex(
    recodex_is_on,
    epoch,
    train_target,
    epoch_trainpreds,
    test_target,
    epoch_testpreds
):
    if recodex_is_on:
        return
    
    print(dedent(
        f"""
        ###
        Epoch #{epoch}:
            -> train RMSE: {sklearn.metrics.root_mean_squared_error(train_target, epoch_trainpreds)}
            -> test RMSE: {sklearn.metrics.root_mean_squared_error(test_target, epoch_testpreds)}
        ###
        """
    )) if epoch % 100 == 0 else None

def print_final_rmse_outside_recodex(recodex_is_on, test_target, predictions):
    if recodex_is_on:
        return

    print(f"RMSE: {sklearn.metrics.root_mean_squared_error(test_target, predictions)}")

class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(
        self,
        name="rental_competition.train.npz",
        url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"
    ):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
            train.data, train.target, random_state=args.seed, test_size=0.25
        )

        all_cols = np.arange(0, train_data.shape[1])
        # Hardcoded just like expected dataset structure
        cat_cols = all_cols[0:8]
        real_cols = all_cols[8:train_data.shape[1]]

        tformer = sklearn.pipeline.Pipeline([
            ('col_tform', sklearn.compose.ColumnTransformer([
                ('category',
                    sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    cat_cols
                ),
                ('std_scale',
                    sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True),
                    real_cols
                )
            ])),
            ('poly_append', sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)),
        ])

        tformed_traind = tformer.fit_transform(train_data)
        traind_rows = tformed_traind.shape[0]
        biased_tf_traind = np.column_stack((tformed_traind, np.ones(traind_rows)))

        tformed_testd = tformer.transform(test_data)
        biased_tf_testd = np.column_stack((tformed_testd, np.ones(tformed_testd.shape[0])))

        generator = np.random.RandomState(args.seed)
        weights = generator.uniform(size=biased_tf_traind.shape[1], low=-0.1, high=0.1)
        for epoch in range(SgdHyperparameters.epochs):
            permutation = generator.permutation(traind_rows)

            for i in range(0, len(permutation), SgdHyperparameters.batch_size):
                batch_indices = permutation[i:max((i + SgdHyperparameters.batch_size), len(permutation))]
                databatch = biased_tf_traind[batch_indices]
                targetbatch = train_target[batch_indices]

                unregularized = np.matmul(
                    np.matmul(databatch, weights) - targetbatch,
                    databatch
                ) / len(batch_indices)
                
                weights_w_zbias = weights.copy()
                weights_w_zbias[-1] = 0
                reg_loss = SgdHyperparameters.l2 * weights_w_zbias

                gradient = unregularized + reg_loss
                weights = weights - SgdHyperparameters.learning_rate * gradient

            epoch_trainpreds = np.matmul(biased_tf_traind, weights)
            epoch_testpreds = np.matmul(biased_tf_testd, weights)

            print_epoch_metrics_outside_recodex(
                args.recodex, epoch,
                train_target, epoch_trainpreds,
                test_target, epoch_testpreds
            )

        model = {
            "transformer": tformer,
            "weights": weights
        }
        predictions = np.matmul(biased_tf_testd, weights)
        print_final_rmse_outside_recodex(args.recodex, test_target, predictions)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)
        test_data = test.data

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # DONE: Generate `predictions` with the test set predictions.
        tformed_testd = model["transformer"].transform(test_data)
        predictions = np.matmul(
            np.column_stack((
                tformed_testd,
                np.ones(tformed_testd.shape[0])
            )),
            model["weights"]
        )

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)