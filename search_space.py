"""
Operator registry and evolutionary search operators.

THIS IS THE FILE THE AGENT EDITS during experiment runs.

It defines:
- PREPARATORS: data preparation steps (imputation, outlier handling)
- PREPROCESSORS: feature transformation steps (scaling, PCA, etc.)
- FEATURE_SELECTORS: feature selection methods
- ALGORITHMS: ML algorithms with hyperparameter ranges
- Evolution parameters (population size, mutation rate, etc.)
- random_pipeline(), mutate(), crossover() — search operators
"""

import copy
import random
import math

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
)
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet,
    LogisticRegression,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from pipeline import PipelineConfig

# ---------------------------------------------------------------------------
# Hyperparameter sampling helpers
# ---------------------------------------------------------------------------

def int_range(low, high):
    """Sample integer uniformly from [low, high]."""
    return random.randint(low, high)

def uniform(low, high):
    """Sample float uniformly from [low, high]."""
    return random.uniform(low, high)

def log_uniform(low, high):
    """Sample float log-uniformly from [low, high]."""
    return math.exp(random.uniform(math.log(low), math.log(high)))

def choice(options):
    """Sample from a list of options."""
    return random.choice(options)

# ---------------------------------------------------------------------------
# Custom transformers for data preparation
# ---------------------------------------------------------------------------

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip outliers to [Q1 - factor*IQR, Q3 + factor*IQR]."""
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.q1_ = np.nanpercentile(X, 25, axis=0)
        self.q3_ = np.nanpercentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64)
        lower = self.q1_ - self.factor * self.iqr_
        upper = self.q3_ + self.factor * self.iqr_
        return np.clip(X, lower, upper)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Winsorize to [lower_pct, upper_pct] percentiles."""
    def __init__(self, lower_pct=5, upper_pct=95):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.lower_ = np.nanpercentile(X, self.lower_pct, axis=0)
        self.upper_ = np.nanpercentile(X, self.upper_pct, axis=0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64)
        return np.clip(X, self.lower_, self.upper_)


# ---------------------------------------------------------------------------
# Preparator registry (imputation, outlier handling)
# ---------------------------------------------------------------------------

PREPARATORS = {
    "SimpleImputer_mean": {
        "class": SimpleImputer,
        "params": {
            "strategy": lambda: "mean",
        },
    },
    "SimpleImputer_median": {
        "class": SimpleImputer,
        "params": {
            "strategy": lambda: "median",
        },
    },
    "SimpleImputer_most_frequent": {
        "class": SimpleImputer,
        "params": {
            "strategy": lambda: "most_frequent",
        },
    },
    "KNNImputer": {
        "class": KNNImputer,
        "params": {
            "n_neighbors": lambda: int_range(3, 15),
        },
    },
    "OutlierClipper": {
        "class": OutlierClipper,
        "params": {
            "factor": lambda: uniform(1.0, 3.0),
        },
    },
    "Winsorizer": {
        "class": Winsorizer,
        "params": {
            "lower_pct": lambda: choice([1, 2, 5]),
            "upper_pct": lambda: choice([95, 98, 99]),
        },
    },
}

# ---------------------------------------------------------------------------
# Preprocessor registry
# ---------------------------------------------------------------------------

PREPROCESSORS = {
    "StandardScaler": {
        "class": StandardScaler,
        "params": {},
    },
    "MinMaxScaler": {
        "class": MinMaxScaler,
        "params": {},
    },
    "RobustScaler": {
        "class": RobustScaler,
        "params": {},
    },
    "PCA": {
        "class": PCA,
        "params": {
            "n_components": lambda: uniform(0.5, 0.99),
        },
    },
    "PolynomialFeatures": {
        "class": PolynomialFeatures,
        "params": {
            "degree": lambda: 2,
            "interaction_only": lambda: True,
            "include_bias": lambda: False,
        },
    },
}

# ---------------------------------------------------------------------------
# Feature selector registry
# ---------------------------------------------------------------------------

FEATURE_SELECTORS = {
    "passthrough": {
        "class": None,
        "params": {},
    },
    "SelectKBest": {
        "class": SelectKBest,
        "score_func": {
            "regression": f_regression,
            "classification": f_classif,
        },
        "params": {
            "k": lambda n_features: max(1, int_range(max(1, n_features // 4), n_features)),
        },
    },
}

# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGORITHMS = {
    "RandomForest": {
        "regressor": RandomForestRegressor,
        "classifier": RandomForestClassifier,
        "params": {
            "n_estimators": lambda: int_range(50, 500),
            "max_depth": lambda: choice([None, 5, 10, 15, 20, 30]),
            "min_samples_split": lambda: int_range(2, 20),
            "min_samples_leaf": lambda: int_range(1, 10),
            "max_features": lambda: choice(["sqrt", "log2", 0.5, 0.8, 1.0]),
        },
    },
    "GradientBoosting": {
        "regressor": GradientBoostingRegressor,
        "classifier": GradientBoostingClassifier,
        "params": {
            "n_estimators": lambda: int_range(50, 500),
            "learning_rate": lambda: log_uniform(0.01, 0.3),
            "max_depth": lambda: int_range(3, 10),
            "min_samples_split": lambda: int_range(2, 20),
            "min_samples_leaf": lambda: int_range(1, 10),
            "subsample": lambda: uniform(0.6, 1.0),
        },
    },
    "ExtraTrees": {
        "regressor": ExtraTreesRegressor,
        "classifier": ExtraTreesClassifier,
        "params": {
            "n_estimators": lambda: int_range(50, 500),
            "max_depth": lambda: choice([None, 5, 10, 15, 20, 30]),
            "min_samples_split": lambda: int_range(2, 20),
            "min_samples_leaf": lambda: int_range(1, 10),
            "max_features": lambda: choice(["sqrt", "log2", 0.5, 0.8, 1.0]),
        },
    },
    "Ridge": {
        "regressor": Ridge,
        "classifier": LogisticRegression,  # Ridge classifier analog
        "params": {
            "alpha": lambda: log_uniform(0.001, 100.0),
        },
    },
    "Lasso": {
        "regressor": Lasso,
        "classifier": LogisticRegression,
        "params": {
            "alpha": lambda: log_uniform(0.001, 10.0),
        },
    },
    "ElasticNet": {
        "regressor": ElasticNet,
        "classifier": LogisticRegression,
        "params": {
            "alpha": lambda: log_uniform(0.001, 10.0),
            "l1_ratio": lambda: uniform(0.1, 0.9),
        },
    },
    "SVR": {
        "regressor": SVR,
        "classifier": SVC,
        "params": {
            "C": lambda: log_uniform(0.01, 100.0),
            "kernel": lambda: choice(["rbf", "linear", "poly"]),
        },
    },
    "KNeighbors": {
        "regressor": KNeighborsRegressor,
        "classifier": KNeighborsClassifier,
        "params": {
            "n_neighbors": lambda: int_range(3, 30),
            "weights": lambda: choice(["uniform", "distance"]),
            "p": lambda: choice([1, 2]),
        },
    },
    "DecisionTree": {
        "regressor": DecisionTreeRegressor,
        "classifier": DecisionTreeClassifier,
        "params": {
            "max_depth": lambda: choice([None, 5, 10, 15, 20, 30]),
            "min_samples_split": lambda: int_range(2, 20),
            "min_samples_leaf": lambda: int_range(1, 10),
        },
    },
    "AdaBoost": {
        "regressor": AdaBoostRegressor,
        "classifier": AdaBoostClassifier,
        "params": {
            "n_estimators": lambda: int_range(50, 300),
            "learning_rate": lambda: log_uniform(0.01, 1.0),
        },
    },
}

# Optional: XGBoost and LightGBM (imported lazily)
try:
    from xgboost import XGBRegressor, XGBClassifier
    ALGORITHMS["XGBoost"] = {
        "regressor": XGBRegressor,
        "classifier": XGBClassifier,
        "params": {
            "n_estimators": lambda: int_range(50, 500),
            "learning_rate": lambda: log_uniform(0.01, 0.3),
            "max_depth": lambda: int_range(3, 10),
            "min_child_weight": lambda: int_range(1, 10),
            "subsample": lambda: uniform(0.6, 1.0),
            "colsample_bytree": lambda: uniform(0.5, 1.0),
            "reg_alpha": lambda: log_uniform(1e-5, 10.0),
            "reg_lambda": lambda: log_uniform(1e-5, 10.0),
            "verbosity": lambda: 0,
        },
    }
except ImportError:
    pass

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    ALGORITHMS["LightGBM"] = {
        "regressor": LGBMRegressor,
        "classifier": LGBMClassifier,
        "params": {
            "n_estimators": lambda: int_range(50, 500),
            "learning_rate": lambda: log_uniform(0.01, 0.3),
            "max_depth": lambda: int_range(3, 15),
            "num_leaves": lambda: int_range(15, 127),
            "min_child_samples": lambda: int_range(5, 50),
            "subsample": lambda: uniform(0.6, 1.0),
            "colsample_bytree": lambda: uniform(0.5, 1.0),
            "reg_alpha": lambda: log_uniform(1e-5, 10.0),
            "reg_lambda": lambda: log_uniform(1e-5, 10.0),
            "verbosity": lambda: -1,
        },
    }
except ImportError:
    pass

# Optional: MLP via PyTorch (wrapped in sklearn API)
try:
    import torch
    from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
    from sklearn.preprocessing import LabelEncoder

    class _TorchMLPBase(BaseEstimator):
        def __init__(self, hidden_sizes=(128, 64), lr=0.001, epochs=100, batch_size=64, dropout=0.1):
            self.hidden_sizes = hidden_sizes
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.dropout = dropout
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        def _build_net(self, input_dim, output_dim):
            layers = []
            prev = input_dim
            for h in self.hidden_sizes:
                layers.extend([
                    torch.nn.Linear(prev, h),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(self.dropout),
                ])
                prev = h
            layers.append(torch.nn.Linear(prev, output_dim))
            return torch.nn.Sequential(*layers).to(self.device)

        def _to_tensor(self, X, dtype=torch.float32):
            if hasattr(X, 'values'):
                X = X.values
            return torch.tensor(X, dtype=dtype, device=self.device)

    class TorchMLPRegressor(_TorchMLPBase, RegressorMixin):
        def fit(self, X, y):
            X_t = self._to_tensor(X)
            y_t = self._to_tensor(y.values if hasattr(y, 'values') else y).unsqueeze(1)
            self.net_ = self._build_net(X_t.shape[1], 1)
            opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
            loss_fn = torch.nn.MSELoss()
            self.net_.train()
            n = X_t.shape[0]
            for epoch in range(self.epochs):
                perm = torch.randperm(n, device=self.device)
                for i in range(0, n, self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    pred = self.net_(X_t[idx])
                    loss = loss_fn(pred, y_t[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            return self

        def predict(self, X):
            X_t = self._to_tensor(X)
            self.net_.eval()
            with torch.no_grad():
                return self.net_(X_t).squeeze(1).cpu().numpy()

    class TorchMLPClassifier(_TorchMLPBase, ClassifierMixin):
        def fit(self, X, y):
            self.le_ = LabelEncoder()
            y_enc = self.le_.fit_transform(y)
            n_classes = len(self.le_.classes_)
            X_t = self._to_tensor(X)
            y_t = torch.tensor(y_enc, dtype=torch.long, device=self.device)
            self.net_ = self._build_net(X_t.shape[1], n_classes)
            opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
            loss_fn = torch.nn.CrossEntropyLoss()
            self.net_.train()
            n = X_t.shape[0]
            for epoch in range(self.epochs):
                perm = torch.randperm(n, device=self.device)
                for i in range(0, n, self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    pred = self.net_(X_t[idx])
                    loss = loss_fn(pred, y_t[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            return self

        def predict(self, X):
            X_t = self._to_tensor(X)
            self.net_.eval()
            with torch.no_grad():
                logits = self.net_(X_t)
                return self.le_.inverse_transform(logits.argmax(dim=1).cpu().numpy())

        def predict_proba(self, X):
            X_t = self._to_tensor(X)
            self.net_.eval()
            with torch.no_grad():
                logits = self.net_(X_t)
                return torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

    ALGORITHMS["MLP"] = {
        "regressor": TorchMLPRegressor,
        "classifier": TorchMLPClassifier,
        "params": {
            "hidden_sizes": lambda: choice([(64, 32), (128, 64), (256, 128), (128, 64, 32), (256, 128, 64)]),
            "lr": lambda: log_uniform(0.0001, 0.01),
            "epochs": lambda: int_range(50, 200),
            "batch_size": lambda: choice([32, 64, 128]),
            "dropout": lambda: uniform(0.0, 0.5),
        },
    }
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Evolution parameters
# ---------------------------------------------------------------------------

POPULATION_SIZE = 20
OFFSPRING_PER_GEN = 10
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.7
MIGRATION_INTERVAL = 15  # seconds between island migrations
MIN_EVALUATIONS = 200    # keep searching past time_budget until this many evals done

# ---------------------------------------------------------------------------
# Build the registry dict used by pipeline.py
# ---------------------------------------------------------------------------

def get_registry():
    """Return the operator registry dict for use by pipeline.build_sklearn_pipeline."""
    return {
        "preparators": PREPARATORS,
        "preprocessors": PREPROCESSORS,
        "feature_selectors": FEATURE_SELECTORS,
        "algorithms": ALGORITHMS,
    }

# ---------------------------------------------------------------------------
# Search operators
# ---------------------------------------------------------------------------

def _sample_params(param_spec, n_features=None):
    """Sample hyperparameters from a param specification dict."""
    params = {}
    for key, sampler in param_spec.items():
        if callable(sampler):
            import inspect
            sig = inspect.signature(sampler)
            if len(sig.parameters) > 0 and n_features is not None:
                params[key] = sampler(n_features)
            else:
                params[key] = sampler()
        else:
            params[key] = sampler
    return params


def random_pipeline(n_features=None):
    """Generate a random PipelineConfig from the search space."""
    # Random preparation: 0-2 steps
    n_data = random.randint(0, 2)
    data_names = random.sample(list(PREPARATORS.keys()), min(n_data, len(PREPARATORS)))
    preparation = []
    for name in data_names:
        params = _sample_params(PREPARATORS[name]["params"], n_features)
        preparation.append((name, params))

    # Random preprocessing: 0-2 steps
    n_prep = random.randint(0, 2)
    prep_names = random.sample(list(PREPROCESSORS.keys()), min(n_prep, len(PREPROCESSORS)))
    preprocessing = []
    for name in prep_names:
        params = _sample_params(PREPROCESSORS[name]["params"], n_features)
        preprocessing.append((name, params))

    # Random feature selection
    fs_name = random.choice(list(FEATURE_SELECTORS.keys()))
    if fs_name == "passthrough":
        feature_selection = ("passthrough", {})
    else:
        params = _sample_params(FEATURE_SELECTORS[fs_name]["params"], n_features)
        feature_selection = (fs_name, params)

    # Random algorithm
    alg_name = random.choice(list(ALGORITHMS.keys()))
    alg_params = _sample_params(ALGORITHMS[alg_name]["params"], n_features)
    algorithm = (alg_name, alg_params)

    return PipelineConfig(
        preparation=preparation,
        preprocessing=preprocessing,
        feature_selection=feature_selection,
        algorithm=algorithm,
    )


def mutate(config: PipelineConfig, n_features=None) -> PipelineConfig:
    """Mutate a pipeline config. Returns a new config (does not modify in place)."""
    config = copy.deepcopy(config)

    mutation_type = random.choice(["hyperparams", "swap_algorithm", "modify_preparation", "modify_preprocessing", "modify_feature_selection"])

    if mutation_type == "hyperparams":
        # Perturb one or more hyperparameters of the algorithm
        alg_name, alg_params = config.algorithm
        if alg_name in ALGORITHMS:
            param_spec = ALGORITHMS[alg_name]["params"]
            # Re-sample 1-2 params
            keys_to_mutate = random.sample(
                list(param_spec.keys()),
                min(random.randint(1, 2), len(param_spec))
            )
            new_params = dict(alg_params)
            for key in keys_to_mutate:
                sampler = param_spec[key]
                if callable(sampler):
                    import inspect
                    sig = inspect.signature(sampler)
                    if len(sig.parameters) > 0 and n_features is not None:
                        new_params[key] = sampler(n_features)
                    else:
                        new_params[key] = sampler()
            config.algorithm = (alg_name, new_params)

    elif mutation_type == "swap_algorithm":
        # Replace the algorithm entirely
        alg_name = random.choice(list(ALGORITHMS.keys()))
        alg_params = _sample_params(ALGORITHMS[alg_name]["params"], n_features)
        config.algorithm = (alg_name, alg_params)

    elif mutation_type == "modify_preparation":
        # Add, remove, or replace a data preparation step
        action = random.choice(["add", "remove", "replace"])
        if action == "add" and len(config.preparation) < 3:
            name = random.choice(list(PREPARATORS.keys()))
            params = _sample_params(PREPARATORS[name]["params"], n_features)
            config.preparation.append((name, params))
        elif action == "remove" and len(config.preparation) > 0:
            idx = random.randint(0, len(config.preparation) - 1)
            config.preparation.pop(idx)
        elif action == "replace" and len(config.preparation) > 0:
            idx = random.randint(0, len(config.preparation) - 1)
            name = random.choice(list(PREPARATORS.keys()))
            params = _sample_params(PREPARATORS[name]["params"], n_features)
            config.preparation[idx] = (name, params)

    elif mutation_type == "modify_preprocessing":
        # Add, remove, or replace a preprocessing step
        action = random.choice(["add", "remove", "replace"])
        if action == "add" and len(config.preprocessing) < 3:
            name = random.choice(list(PREPROCESSORS.keys()))
            params = _sample_params(PREPROCESSORS[name]["params"], n_features)
            config.preprocessing.append((name, params))
        elif action == "remove" and len(config.preprocessing) > 0:
            idx = random.randint(0, len(config.preprocessing) - 1)
            config.preprocessing.pop(idx)
        elif action == "replace" and len(config.preprocessing) > 0:
            idx = random.randint(0, len(config.preprocessing) - 1)
            name = random.choice(list(PREPROCESSORS.keys()))
            params = _sample_params(PREPROCESSORS[name]["params"], n_features)
            config.preprocessing[idx] = (name, params)

    elif mutation_type == "modify_feature_selection":
        fs_name = random.choice(list(FEATURE_SELECTORS.keys()))
        if fs_name == "passthrough":
            config.feature_selection = ("passthrough", {})
        else:
            params = _sample_params(FEATURE_SELECTORS[fs_name]["params"], n_features)
            config.feature_selection = (fs_name, params)

    return config


def crossover(parent_a: PipelineConfig, parent_b: PipelineConfig) -> PipelineConfig:
    """
    Block swap crossover: swap entire pipeline stages between parents.
    Returns a single offspring.
    """
    child = copy.deepcopy(parent_a)

    # Each block has independent probability of being swapped from parent_b
    if random.random() < 0.5:
        child.preparation = copy.deepcopy(parent_b.preparation)

    if random.random() < 0.5:
        child.preprocessing = copy.deepcopy(parent_b.preprocessing)

    if random.random() < 0.5:
        child.feature_selection = copy.deepcopy(parent_b.feature_selection)

    if random.random() < 0.5:
        child.algorithm = copy.deepcopy(parent_b.algorithm)

    return child
