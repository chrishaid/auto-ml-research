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
    if cls is None:
        raise ValueError(f"Algorithm '{alg_name}' has no {cls_key} implementation")
    # Filter params to only those the constructor accepts (handles shared param dicts
    # where regressor and classifier have different valid params)
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    # If **kwargs is accepted, pass everything
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_kwargs:
        filtered_params = alg_params
    else:
        filtered_params = {k: v for k, v in alg_params.items() if k in valid_params}
    steps.append(("algorithm", cls(**filtered_params)))

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
            except (AttributeError, NotImplementedError, IndexError):
                # Fallback: decision_function for models without predict_proba
                algo = pipe.named_steps.get("algorithm")
                if hasattr(algo, "decision_function"):
                    y_pred = pipe.decision_function(X_val)
                else:
                    y_pred = pipe.predict(X_val)
        else:
            y_pred = pipe.predict(X_val)
        elapsed = time.time() - t0
        score = evaluate(y_val, y_pred, metric_name, task_type)
        return (score, elapsed)
    except Exception:
        return (None, traceback.format_exc())


def _subprocess_worker(conn, config_dict, task_type, X_train, y_train, X_val, y_val, metric_name):
    """Run in a child process. Reconstructs registry locally to avoid pickling lambdas."""
    try:
        from search_space import get_registry
        from prepare import evaluate
        registry = get_registry()
        cfg = PipelineConfig.from_dict(config_dict)
        pipe = build_sklearn_pipeline(cfg, registry, task_type)
        fit_t0 = time.time()
        pipe.fit(X_train, y_train)
        if metric_name in ("auc", "logloss"):
            try:
                y_pred = pipe.predict_proba(X_val)[:, 1]
            except (AttributeError, NotImplementedError, IndexError):
                # Fallback: decision_function for models without predict_proba
                algo = pipe.named_steps.get("algorithm")
                if hasattr(algo, "decision_function"):
                    y_pred = pipe.decision_function(X_val)
                else:
                    y_pred = pipe.predict(X_val)
        else:
            y_pred = pipe.predict(X_val)
        elapsed = time.time() - fit_t0
        score = evaluate(y_val, y_pred, metric_name, task_type)
        conn.send((score, elapsed, None))
    except Exception:
        conn.send((None, 0.0, traceback.format_exc()))
    finally:
        conn.close()


def execute_pipeline(config: PipelineConfig, registry: dict, task_type: str,
                     X_train, y_train, X_val, y_val, metric_name: str,
                     timeout: int = 60):
    """
    Fit and evaluate a pipeline with timeout protection via subprocess.
    Uses multiprocessing so timed-out work is actually killed (not left as
    a zombie daemon thread consuming CPU).
    Returns (score, elapsed_seconds, error_string_or_None).
    """
    import multiprocessing as mp

    t0 = time.time()
    parent_conn, child_conn = mp.Pipe(duplex=False)

    proc = mp.Process(
        target=_subprocess_worker,
        args=(child_conn, config.to_dict(), task_type,
              X_train, y_train, X_val, y_val, metric_name),
    )
    proc.start()
    child_conn.close()  # parent doesn't write

    # Wait for result with timeout
    if parent_conn.poll(timeout):
        try:
            score, elapsed, error = parent_conn.recv()
        except EOFError:
            score, elapsed, error = None, time.time() - t0, "process crashed"
    else:
        score, elapsed, error = None, time.time() - t0, "timeout"

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=2)
    else:
        proc.join(timeout=2)

    parent_conn.close()
    return (score, elapsed, error)
