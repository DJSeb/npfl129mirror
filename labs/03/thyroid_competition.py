#!/usr/bin/env python3

# Me: 4ae50bd7-2516-11ec-986f-f39926f24a9c (David jaromír Šebánek)
# Team member 1: a652f90b-2220-11ec-986f-f39926f24a9c (Norbert Horváth)
# Team member 2: 5212d2e4-2516-11ec-986f-f39926f24a9c (Samuel Serafín)

import argparse
from dataclasses import dataclass
import lzma
import os
from pathlib import Path
import pickle
import sklearn
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt

# Just to turn off probably unimportant warnings.
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=69, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path",
    default=str((Path(__file__).resolve().parent / "thyroid_competition.model")),
    type=str,
    help="Model path"
)

@dataclass(frozen=True)
class Hyperparameters:
    @dataclass(frozen=True)
    class DatasetSplit:
        testset_ratio = 0.3

    @dataclass(frozen=True)
    class Preprocessing:
        # One-hot
        onehot_is_sparse = False
        unknown_handling = 'ignore'
        # Standard scaler
        std_scale_with_mean = True
        std_scale_with_std = True
        # Polynomial features
        poly_degree = 2
        poly_bias = False

    @dataclass(frozen=True)
    class CrossValidation:
        n_folds = 5
        cpu_parallelization = -1 # Use all
        average_all_folds = True

    @dataclass(frozen=True)
    class SklearnLogRess:
        reg_strengths = 10 # Usually called Cs
        l1_ratios = (0.0,) # Just use L2 reg (between 0 and 1 for elastic net, 1 for L1), cross-val'd
        have_internal_bias = True
        use_dual = False # For liblinear + L2, works good if n_samples <= n_features
        scoring_goal = 'accuracy' # Per task's description
        # liblinear is good for binary classification (which we are doing)
        # sag is Stochastic Average Gradient Descent (SAGD) (this one needs scaling preproc. for fast convergence)
        solver = 'liblinear'
        tolerance = 1e-4
        epochs = 200
        class_weights = None
        bias_value = 1
        # Sklearn-special, False from 1.10 (to be deprecated)
        legacy_mode_on = False

def print_accuracy_outside_recodex(recodex_is_on, target, predictions, prefix=""):
    if recodex_is_on:
        return

    print(f"{prefix} Accuracy: {sklearn.metrics.accuracy_score(target, predictions)}")

def print_log_loss_outside_recodex(recodex_is_on, target, probs, prefix=""):
    if recodex_is_on:
        return

    print(f"{prefix} Log Loss: {sklearn.metrics.log_loss(target, probs)}")

class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
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

        # DONE: Train a model on the given dataset and store it in `model`.
        ## Split to train/test
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
            train.data, train.target,
            random_state=args.seed,
            test_size=Hyperparameters.DatasetSplit.testset_ratio
        )
        
        ## Classify column types
        n_categories = 15 # As shape of data dictates
        all_cols = np.arange(0, train_data.shape[1])
        cat_cols = all_cols[0 : n_categories] 
        real_cols = all_cols[n_categories : train_data.shape[1]]

        ## Apply preprocessing
        pproc = sklearn.pipeline.Pipeline((
            ("col_tform", sklearn.compose.ColumnTransformer([
                ("category",
                    sklearn.preprocessing.OneHotEncoder(
                        sparse_output=Hyperparameters.Preprocessing.onehot_is_sparse,
                        handle_unknown=Hyperparameters.Preprocessing.unknown_handling
                    ),
                    cat_cols),
                ("std_scale",
                    sklearn.preprocessing.StandardScaler(
                        with_mean=Hyperparameters.Preprocessing.std_scale_with_mean,
                        with_std=Hyperparameters.Preprocessing.std_scale_with_std
                    ),
                    real_cols),
                ])
            ),
            ("poly_append", sklearn.preprocessing.PolynomialFeatures(
                Hyperparameters.Preprocessing.poly_degree,
                include_bias=Hyperparameters.Preprocessing.poly_bias)
            ),
        ))

        ## Configure predictor
        predictor = sklearn.linear_model.LogisticRegressionCV(
            # General arg (runtime given)
            random_state=args.seed,
            # CV arg
            cv=Hyperparameters.CrossValidation.n_folds,
            n_jobs=Hyperparameters.CrossValidation.cpu_parallelization,
            refit=Hyperparameters.CrossValidation.average_all_folds,
            # Inner model arg.
            Cs=Hyperparameters.SklearnLogRess.reg_strengths,
            l1_ratios=Hyperparameters.SklearnLogRess.l1_ratios,
            fit_intercept=Hyperparameters.SklearnLogRess.have_internal_bias,
            dual=Hyperparameters.SklearnLogRess.use_dual,
            scoring=Hyperparameters.SklearnLogRess.scoring_goal,
            solver=Hyperparameters.SklearnLogRess.solver,
            tol=Hyperparameters.SklearnLogRess.tolerance,
            max_iter=Hyperparameters.SklearnLogRess.epochs,
            class_weight=Hyperparameters.SklearnLogRess.class_weights,
            intercept_scaling=Hyperparameters.SklearnLogRess.bias_value,
            use_legacy_attributes=Hyperparameters.SklearnLogRess.legacy_mode_on
        )

        ## Assemble to one model
        model = sklearn.pipeline.Pipeline((
            ('preprocessing', pproc),
            ('predicting_model', predictor),
        ))

        ## Train check
        model.fit(train_data, train_target)
        print_accuracy_outside_recodex(args.recodex, train_target, model.predict(train_data), 'Train')
        print_log_loss_outside_recodex(args.recodex, train_target, model.predict_proba(train_data), 'Train')

        ## Test check
        print_accuracy_outside_recodex(args.recodex, test_target, model.predict(test_data), 'Test')
        print_log_loss_outside_recodex(args.recodex, test_target, model.predict_proba(test_data), 'Test')

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # DONE: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
