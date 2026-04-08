"""
Data loading, splitting, and evaluation harness for AutoML evolution.

This file is READ-ONLY during experiment runs. It provides:
- Problem definition loading from problem.toml
- Data loading from files (CSV/Parquet), sklearn datasets, or Snowflake
- Train/val splitting (stratified for classification)
- Metric evaluation with direction awareness
- Score-to-fitness normalization (higher = better)

Usage:
    from prepare import load_problem, load_data, split_data, evaluate, score_to_fitness
"""

import os
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

METRICS = {
    # Regression
    "rmse":     {"direction": "minimize", "fn": "_rmse"},
    "mse":      {"direction": "minimize", "fn": "_mse"},
    "mae":      {"direction": "minimize", "fn": "_mae"},
    "r2":       {"direction": "maximize", "fn": "_r2"},
    "adj_r2":   {"direction": "maximize", "fn": "_adj_r2"},
    # Classification
    "auc":      {"direction": "maximize", "fn": "_auc"},
    "logloss":  {"direction": "minimize", "fn": "_logloss"},
    "accuracy": {"direction": "maximize", "fn": "_accuracy"},
    "f1":       {"direction": "maximize", "fn": "_f1"},
}


def _rmse(y_true, y_pred, **kw):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def _mse(y_true, y_pred, **kw):
    return np.mean((y_true - y_pred) ** 2)

def _mae(y_true, y_pred, **kw):
    return np.mean(np.abs(y_true - y_pred))

def _r2(y_true, y_pred, **kw):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

def _adj_r2(y_true, y_pred, n_features=1, **kw):
    r2 = _r2(y_true, y_pred)
    n = len(y_true)
    p = n_features
    if n - p - 1 <= 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def _auc(y_true, y_pred, **kw):
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.0

def _logloss(y_true, y_pred, **kw):
    from sklearn.metrics import log_loss
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return log_loss(y_true, y_pred)

def _accuracy(y_true, y_pred, **kw):
    from sklearn.metrics import accuracy_score
    # Round predictions for classification
    if y_pred.dtype.kind == 'f':
        y_pred = np.round(y_pred)
    return accuracy_score(y_true, y_pred)

def _f1(y_true, y_pred, **kw):
    from sklearn.metrics import f1_score
    if y_pred.dtype.kind == 'f':
        y_pred = np.round(y_pred)
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)

_METRIC_FNS = {
    "_rmse": _rmse, "_mse": _mse, "_mae": _mae, "_r2": _r2, "_adj_r2": _adj_r2,
    "_auc": _auc, "_logloss": _logloss, "_accuracy": _accuracy, "_f1": _f1,
}

# ---------------------------------------------------------------------------
# Problem loading
# ---------------------------------------------------------------------------

def load_problem(path="problem.toml"):
    """Parse problem.toml and return a dict with all settings."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    problem = raw["problem"]
    data = raw["data"]
    budget = raw.get("budget", {})

    metric = problem["metric"]
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Available: {list(METRICS.keys())}")

    direction = problem.get("direction", METRICS[metric]["direction"])

    n_workers = budget.get("n_workers", max(1, (os.cpu_count() or 2) // 2))
    n_workers = min(n_workers, 8)

    return {
        "name": problem["name"],
        "task": problem["task"],
        "metric": metric,
        "direction": direction,
        "data_source": data.get("source", "file"),
        "data_config": data,
        "time_budget": budget.get("time_budget", 300),
        "pipeline_timeout": budget.get("pipeline_timeout", 60),
        "n_workers": n_workers,
        "min_evaluations": budget.get("min_evaluations", 200),
        "max_rows": budget.get("max_rows", None),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(problem):
    """Load data based on problem config. Returns (X: DataFrame, y: Series)."""
    source = problem["data_source"]
    data_config = problem["data_config"]

    if source == "file":
        X, y = _load_from_file(data_config)
    elif source == "sklearn":
        X, y = _load_from_sklearn(data_config)
    elif source == "snowflake":
        X, y = _load_from_snowflake(data_config)
    else:
        raise ValueError(f"Unknown data source: {source}")

    max_rows = problem.get("max_rows")
    if max_rows and len(X) > max_rows:
        print(f"Subsampling from {len(X)} to {max_rows} rows (stratified)...")
        is_clf = y.dtype == object or y.dtype.name == "category" or y.nunique() < 20
        X, _, y, _ = train_test_split(
            X, y, train_size=max_rows, random_state=42,
            stratify=y if is_clf else None,
        )

    return X, y


def _load_from_file(config):
    """Load from CSV or Parquet file."""
    path = config["train"]
    target = config["target"]
    id_column = config.get("id_column")

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if id_column and id_column in df.columns:
        df = df.drop(columns=[id_column])

    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def _load_from_sklearn(config):
    """Load a built-in sklearn dataset."""
    from sklearn import datasets
    dataset_name = config["dataset"]
    loader = getattr(datasets, f"fetch_{dataset_name}", None)
    if loader is None:
        loader = getattr(datasets, f"load_{dataset_name}", None)
    if loader is None:
        raise ValueError(f"Unknown sklearn dataset: {dataset_name}")

    data = loader(as_frame=True)
    X = data.data
    target = config.get("target", data.target.name if hasattr(data.target, 'name') else "target")
    y = data.target
    return X, y


def _load_from_snowflake(config):
    """Load data from Snowflake query, cache as local parquet."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "snowflake_cache")
    os.makedirs(cache_dir, exist_ok=True)

    query = config["query"]
    target = config["target"]
    id_column = config.get("id_column")
    connection = config.get("connection", "default")

    # Cache key based on query hash
    import hashlib
    query_hash = hashlib.md5(query.encode()).hexdigest()[:12]
    cache_path = os.path.join(cache_dir, f"{query_hash}.parquet")

    if os.path.exists(cache_path):
        print(f"Loading cached Snowflake data from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print(f"Querying Snowflake (connection={connection})...")
        try:
            import snowflake.connector
        except ImportError:
            raise ImportError("snowflake-connector-python required for Snowflake data source. "
                              "Install with: pip install snowflake-connector-python")

        # Use default connection from ~/.snowflake/config.toml
        # Pass private key passphrase from env if set (for encrypted keys)
        connect_kwargs = {"connection_name": connection}
        passphrase = os.environ.get("PRIVATE_KEY_PASSPHRASE")
        if passphrase:
            connect_kwargs["private_key_file_pwd"] = passphrase
        conn = snowflake.connector.connect(**connect_kwargs)
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            df = cursor.fetch_pandas_all()
        finally:
            conn.close()

        df.to_parquet(cache_path)
        print(f"Cached Snowflake data to {cache_path}")

    if id_column and id_column in df.columns:
        df = df.drop(columns=[id_column])

    y = df[target]
    X = df.drop(columns=[target])
    return X, y


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_data(X, y, val_ratio=0.2, seed=42):
    """Split into train/val. Stratified for classification tasks."""
    # Detect if classification based on target dtype
    is_classification = y.dtype == object or y.dtype.name == "category" or y.nunique() < 20

    stratify = y if is_classification else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=seed, stratify=stratify
    )
    return X_train, X_val, y_train, y_val


# ---------------------------------------------------------------------------
# Auto preprocessing
# ---------------------------------------------------------------------------

def auto_preprocess(X):
    """
    Basic preprocessing: encode categoricals, fill missing values.
    Returns preprocessed DataFrame.
    """
    X = X.copy()

    # Identify column types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Fill missing numerics with median
    for col in num_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Encode categoricals with label encoding
    for col in cat_cols:
        X[col] = X[col].fillna("__missing__")
        X[col] = X[col].astype("category").cat.codes

    return X


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, metric_name, task_type="regression", n_features=1):
    """Compute metric score. Returns a float."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        # Return worst possible score
        direction = METRICS[metric_name]["direction"]
        return float("inf") if direction == "minimize" else float("-inf")

    fn_name = METRICS[metric_name]["fn"]
    fn = _METRIC_FNS[fn_name]
    return fn(y_true, y_pred, n_features=n_features)


def score_to_fitness(score, metric_name):
    """Normalize score so that higher = better (for evolution selection)."""
    direction = METRICS[metric_name]["direction"]
    if direction == "maximize":
        return score
    else:
        return -score


def get_metric_direction(metric_name):
    """Return 'minimize' or 'maximize'."""
    return METRICS[metric_name]["direction"]


# ---------------------------------------------------------------------------
# Main (standalone test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    problem_path = sys.argv[1] if len(sys.argv) > 1 else "problem.toml"
    problem = load_problem(problem_path)
    print(f"Problem: {problem['name']} ({problem['task']})")
    print(f"Metric: {problem['metric']} ({problem['direction']})")

    X, y = load_data(problem)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Features: {list(X.columns)}")

    X = auto_preprocess(X)
    X_train, X_val, y_train, y_val = split_data(X, y)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Quick sanity check with a simple model
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    if problem["task"] == "regression":
        model = RandomForestRegressor(n_estimators=10, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = evaluate(y_val, y_pred, problem["metric"], problem["task"])
    print(f"Sanity check score ({problem['metric']}): {score:.6f}")
