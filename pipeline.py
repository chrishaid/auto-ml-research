"""
Pipeline configuration, construction, and execution.

A pipeline is a linear chain: preparation -> preprocessing -> feature_selection -> algorithm.
Each stage is represented as a (name, {params}) tuple.
PipelineConfig is the unit of evolution — it gets mutated, crossed over, and evaluated.
"""

import time
import traceback
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout

import numpy as np
from sklearn.pipeline import Pipeline


def _describe_step(name, params):
    """Format a single pipeline step as a human-readable string."""
    if params:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{name}({param_str})"
    return name


@dataclass
class PipelineConfig:
    """Serializable pipeline specification."""
    preparation: list[tuple[str, dict]] = field(default_factory=list)
    preprocessing: list[tuple[str, dict]] = field(default_factory=list)
    feature_selection: tuple[str, dict] = field(default_factory=lambda: ("passthrough", {}))
    algorithm: tuple[str, dict] = field(default_factory=lambda: ("RandomForestRegressor", {}))

    def to_dict(self):
        return {
            "preparation": [(n, dict(p)) for n, p in self.preparation],
            "preprocessing": [(n, dict(p)) for n, p in self.preprocessing],
            "feature_selection": (self.feature_selection[0], dict(self.feature_selection[1])),
            "algorithm": (self.algorithm[0], dict(self.algorithm[1])),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            preparation=[(n, p) for n, p in d.get("preparation", [])],
            preprocessing=[(n, p) for n, p in d["preprocessing"]],
            feature_selection=tuple(d["feature_selection"]),
            algorithm=tuple(d["algorithm"]),
        )

    def describe(self):
        parts = []
        for name, params in self.preparation:
            parts.append(_describe_step(name, params))
        for name, params in self.preprocessing:
            parts.append(_describe_step(name, params))
        fs_name, fs_params = self.feature_selection
        if fs_name != "passthrough":
            parts.append(_describe_step(fs_name, fs_params))
        parts.append(_describe_step(*self.algorithm))
        return " -> ".join(parts) if parts else "empty"


def build_sklearn_pipeline(config: PipelineConfig, registry: dict, task_type: str) -> Pipeline:
    """Instantiate an sklearn Pipeline from a PipelineConfig using the operator registry."""
    steps = []

    for i, (name, params) in enumerate(config.preparation):
        entry = registry["preparators"][name]
        cls = entry["class"]
        steps.append((f"data_{i}_{name}", cls(**params)))

    for i, (name, params) in enumerate(config.preprocessing):
        entry = registry["preprocessors"][name]
        cls = entry["class"]
        steps.append((f"prep_{i}_{name}", cls(**params)))

    fs_name, fs_params = config.feature_selection
    if fs_name != "passthrough":
        entry = registry["feature_selectors"][fs_name]
        cls = entry["class"]
        # Some selectors need score_func
        init_params = dict(fs_params)
        if "score_func" in entry:
            init_params.setdefault("score_func", entry["score_func"][task_type])
        steps.append(("feature_selection", cls(**init_params)))

    alg_name, alg_params = config.algorithm
    entry = registry["algorithms"][alg_name]
    cls_key = "regressor" if task_type == "regression" else "classifier"
    cls = entry[cls_key]
    steps.append(("algorithm", cls(**alg_params)))

    return Pipeline(steps) if steps else Pipeline([("passthrough", "passthrough")])


def _run_pipeline(config_dict, registry, task_type, X_train, y_train, X_val, y_val, metric_name):
    """Worker function for subprocess execution. Returns (score, elapsed) or (None, error_str)."""
    from prepare import evaluate  # import here to avoid pickling issues
    try:
        config = PipelineConfig.from_dict(config_dict)
        pipe = build_sklearn_pipeline(config, registry, task_type)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        if metric_name in ("auc", "logloss"):
            try:
                y_pred = pipe.predict_proba(X_val)[:, 1]
            except (AttributeError, NotImplementedError):
                y_pred = pipe.predict(X_val)
        else:
            y_pred = pipe.predict(X_val)
        elapsed = time.time() - t0
        score = evaluate(y_val, y_pred, metric_name, task_type)
        return (score, elapsed)
    except Exception:
        return (None, traceback.format_exc())


def execute_pipeline(config: PipelineConfig, registry: dict, task_type: str,
                     X_train, y_train, X_val, y_val, metric_name: str,
                     timeout: int = 60):
    """
    Fit and evaluate a pipeline with timeout protection via threading.
    Returns (score, elapsed_seconds, error_string_or_None).
    """
    import threading

    result_holder = [None, 0.0, None]  # score, elapsed, error

    def _worker():
        try:
            pipe = build_sklearn_pipeline(config, registry, task_type)
            t0 = time.time()
            pipe.fit(X_train, y_train)
            if metric_name in ("auc", "logloss"):
                try:
                    y_pred = pipe.predict_proba(X_val)[:, 1]
                except (AttributeError, NotImplementedError):
                    y_pred = pipe.predict(X_val)
            else:
                y_pred = pipe.predict(X_val)
            elapsed = time.time() - t0
            from prepare import evaluate
            score = evaluate(y_val, y_pred, metric_name, task_type)
            result_holder[0] = score
            result_holder[1] = elapsed
        except Exception:
            result_holder[2] = traceback.format_exc()

    thread = threading.Thread(target=_worker, daemon=True)
    t0 = time.time()
    thread.start()
    thread.join(timeout=timeout)
    elapsed = time.time() - t0

    if thread.is_alive():
        # Thread still running — treat as timeout (daemon thread will be cleaned up)
        return (None, elapsed, "timeout")

    return (result_holder[0], result_holder[1], result_holder[2])
