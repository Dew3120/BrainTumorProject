import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_r06_nonlinear_comparison import (
    NONLINEAR_MODELS,
    make_nonlinear_model,
    model_gate,
)


def minimal_config():
    return {
        "models": {
            "xgboost": {
                "class": "xgboost.XGBClassifier",
                "scale_features": False,
                "n_estimators": 200,
                "max_depth": 3,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "gamma": 0,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 1,
            },
            "mlp": {
                "class": "sklearn.neural_network.MLPClassifier",
                "scale_features": True,
                "hidden_layer_sizes": [32, 32],
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.0001,
                "batch_size": 64,
                "learning_rate": "constant",
                "learning_rate_init": 0.001,
                "max_iter": 1000,
                "shuffle": True,
                "random_state": 42,
                "tol": 0.0001,
                "early_stopping": False,
                "n_iter_no_change": 50,
            },
            "random_forest": {
                "class": "sklearn.ensemble.RandomForestClassifier",
                "scale_features": False,
                "n_estimators": 100,
                "criterion": "gini",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True,
                "class_weight": None,
                "random_state": 42,
                "n_jobs": 1,
            },
        }
    }


def test_r06_gate_includes_exact_boundary():
    assert model_gate(0.975, 0.975)
    assert not model_gate(np.nextafter(0.975, 0.0), 0.975)


def test_r06_model_order_is_locked():
    assert NONLINEAR_MODELS == ("xgboost", "mlp", "random_forest")


def test_r06_builds_all_registered_models():
    config = minimal_config()
    xgb = make_nonlinear_model("xgboost", config)
    mlp = make_nonlinear_model("mlp", config)
    forest = make_nonlinear_model("random_forest", config)
    assert xgb.n_estimators == 200 and xgb.max_depth == 3
    assert mlp.named_steps["estimator"].hidden_layer_sizes == (32, 32)
    assert forest.n_estimators == 100 and forest.min_samples_leaf == 2
