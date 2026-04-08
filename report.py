"""
Post-run report generator for AutoML evolution results.

Usage:
  uv run report.py problem.toml                       # single problem report
  uv run report.py p1.toml p2.toml p3.toml             # combined tabbed report
  uv run report.py --all-problems                       # all .toml files in problems/

Generates a comprehensive report printed to stdout with ANSI colors,
and saves an HTML report as report_{problem_name}.html (single) or
report_combined.html (multi-problem) with embedded charts.
"""

import os
import sys
import csv
import re
import glob as globmod
import warnings
import time
from collections import defaultdict, Counter

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

class C:
    """ANSI escape codes for terminal colors."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    BG_BLUE = "\033[44m"

    @staticmethod
    def colored(text, *codes):
        return "".join(codes) + str(text) + C.RESET


# ---------------------------------------------------------------------------
# Results parsing
# ---------------------------------------------------------------------------

RESULTS_FILE = "results.tsv"


def load_results():
    """Load all rows from results.tsv, return list of dicts."""
    rows = []
    with open(RESULTS_FILE, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def detect_runs(rows):
    """Detect multiple runs by generation resets. Returns list of (start_idx, end_idx)."""
    runs = []
    run_start = 0
    prev_gen = -1
    for i, row in enumerate(rows):
        try:
            g = int(row["generation"])
        except (ValueError, KeyError):
            continue
        if g < prev_gen and i > 0:
            runs.append((run_start, i))
            run_start = i
        prev_gen = g
    runs.append((run_start, len(rows)))
    return runs


def pick_run_for_problem(rows, runs, problem):
    """Pick the run that matches the problem.

    Strategy: retrain a tiny model from the run's best pipeline on the problem's data.
    If the score roughly matches the best score in the run, it's a match.

    Falls back to matching by run index if --run N is given via env var,
    or by best-score proximity to what we'd expect from the problem's data.
    """
    import os

    # Allow explicit run selection via env var or argv
    for arg in sys.argv:
        if arg.startswith("--run="):
            try:
                idx = int(arg.split("=")[1]) - 1
                if 0 <= idx < len(runs):
                    return runs[idx]
            except (ValueError, IndexError):
                pass

    # Match by loading the problem data, training the best pipeline from each run,
    # and seeing which run's best score is reproducible on this problem's data.
    # This is expensive, so instead use a faster heuristic:
    #
    # The runs were executed in a known order. If we can identify which runs
    # are "new" (propensity) vs "old" (california housing), we map by order.
    #
    # Propensity runs: classification (AUC 0.8-0.9 range)
    # California housing: regression (RMSE-based fitness, negative values)

    problem_name = problem["name"]
    task = problem["task"]

    # Filter runs to those matching the task type by checking fitness values
    # Classification runs have positive fitness (AUC), regression have negative (negated RMSE)
    matching_runs = []
    for run_bounds in runs:
        start, end = run_bounds
        run_rows = rows[start:end]
        ok_rows = [r for r in run_rows if r.get("status") == "ok"]
        if not ok_rows:
            continue
        try:
            scores = [float(r["fitness"]) for r in ok_rows if r["fitness"] != "N/A"]
            if not scores:
                continue
            avg_fitness = sum(scores) / len(scores)
        except (ValueError, KeyError):
            continue

        if task == "classification" and avg_fitness > 0:
            matching_runs.append(run_bounds)
        elif task == "regression" and avg_fitness < 0:
            matching_runs.append(run_bounds)

    if not matching_runs:
        return runs[-1]

    # Among matching runs, use the last N runs (most recent execution batch).
    # The propensity problems are run in order: email, event, web
    PROBLEM_ORDER = ["email-propensity", "event-propensity", "web-propensity"]
    if problem_name in PROBLEM_ORDER:
        idx = PROBLEM_ORDER.index(problem_name)
        n_problems = len(PROBLEM_ORDER)
        # Take the last batch of runs
        if len(matching_runs) >= n_problems:
            batch = matching_runs[-n_problems:]
            if idx < len(batch):
                return batch[idx]

    return matching_runs[-1]


def parse_description(desc):
    """Parse a pipeline description string into components.

    Returns dict with keys: island, steps (list of step strings),
    algorithm (name string), full_desc (without island tag).
    """
    result = {"island": None, "steps": [], "algorithm": None, "full_desc": desc, "error": None}

    # Strip island tag
    m = re.match(r"\[I(\d+)\]\s*(.*)", desc)
    if m:
        result["island"] = int(m.group(1))
        desc_clean = m.group(2)
    else:
        desc_clean = desc

    # Strip error suffix
    if " | " in desc_clean:
        desc_clean, result["error"] = desc_clean.rsplit(" | ", 1)

    result["full_desc"] = desc_clean.strip()

    # Split by " -> "
    parts = [p.strip() for p in desc_clean.split(" -> ") if p.strip()]
    result["steps"] = parts

    if parts:
        result["algorithm"] = parts[-1]

    return result


def extract_algorithm_name(alg_step):
    """Extract just the algorithm name from a step like 'XGBoost(n_estimators=100, ...)'."""
    if not alg_step:
        return "Unknown"
    m = re.match(r"(\w+)", alg_step)
    return m.group(1) if m else alg_step


def extract_block_signature(parsed):
    """Return a simplified tuple (prep_steps, preproc_steps, fs_step, alg_name)
    for tracking block-level changes."""
    steps = parsed["steps"]
    if not steps:
        return ("", "", "", "")

    # The last step is the algorithm
    alg = extract_algorithm_name(steps[-1])

    # Classify each non-algorithm step
    KNOWN_PREPARATORS = {
        "SimpleImputer_mean", "SimpleImputer_median", "SimpleImputer_most_frequent",
        "KNNImputer", "OutlierClipper", "Winsorizer",
    }
    KNOWN_PREPROCESSORS = {
        "StandardScaler", "MinMaxScaler", "RobustScaler", "PCA", "PolynomialFeatures",
    }
    KNOWN_SELECTORS = {"SelectKBest"}

    preps, preprocs, fs = [], [], []
    for step in steps[:-1]:
        name = extract_algorithm_name(step)
        if name in KNOWN_PREPARATORS:
            preps.append(name)
        elif name in KNOWN_PREPROCESSORS:
            preprocs.append(name)
        elif name in KNOWN_SELECTORS:
            fs.append(name)
        else:
            preprocs.append(name)  # default bucket

    return (
        "+".join(preps) if preps else "-",
        "+".join(preprocs) if preprocs else "-",
        "+".join(fs) if fs else "-",
        alg,
    )


# ---------------------------------------------------------------------------
# Section 1: Best Model Summary
# ---------------------------------------------------------------------------

def find_best_row(rows, direction):
    """Find the row with the best score among 'ok' rows."""
    best = None
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            score = float(row["score"])
        except (ValueError, KeyError):
            continue
        if best is None:
            best = row
        else:
            best_score = float(best["score"])
            if direction == "maximize" and score > best_score:
                best = row
            elif direction == "minimize" and score < best_score:
                best = row
    return best


def section_best_model(rows, problem, best_row):
    """Print Section 1: Best Model Summary."""
    direction = problem["direction"]
    metric = problem["metric"]

    score = float(best_row["score"])
    gen = best_row["generation"]
    elapsed = best_row.get("elapsed_s", "?")
    desc = best_row["description"]

    # Count stats
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    err_rows = [r for r in rows if r.get("status") != "ok"]
    gens = set()
    for r in rows:
        try:
            gens.add(int(r["generation"]))
        except (ValueError, KeyError):
            pass
    max_gen = max(gens) if gens else 0

    parsed = parse_description(desc)

    lines = []
    lines.append("")
    lines.append(C.colored("=" * 100, C.BOLD, C.CYAN))
    lines.append(C.colored("  AUTOML EVOLUTION REPORT", C.BOLD, C.WHITE, C.BG_BLUE))
    lines.append(C.colored(f"  Problem: {problem['name']}  |  Task: {problem['task']}  |  Metric: {metric} ({direction})", C.BOLD, C.WHITE, C.BG_BLUE))
    lines.append(C.colored("=" * 100, C.BOLD, C.CYAN))
    lines.append("")
    lines.append(C.colored("--- 1. Best Model Summary ---", C.BOLD, C.YELLOW))
    lines.append("")
    lines.append(f"  {C.BOLD}Best {metric}:{C.RESET}         {C.colored(f'{score:.6f}', C.BOLD, C.GREEN)}")
    lines.append(f"  {C.BOLD}Generations:{C.RESET}        {max_gen}")
    lines.append(f"  {C.BOLD}Total evaluations:{C.RESET}  {len(rows)} ({len(ok_rows)} ok, {len(err_rows)} errors)")
    lines.append(f"  {C.BOLD}Found at gen:{C.RESET}       {gen}")
    lines.append(f"  {C.BOLD}Training time:{C.RESET}      {elapsed}s")
    lines.append("")
    lines.append(f"  {C.BOLD}Pipeline:{C.RESET}")
    for i, step in enumerate(parsed["steps"]):
        arrow = "    " if i == 0 else " -> "
        lines.append(f"  {C.DIM}{arrow}{C.RESET}{C.colored(step, C.CYAN)}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 2: Variable Importance
# ---------------------------------------------------------------------------

def compute_feature_importances(best_row, problem):
    """Retrain the best pipeline on full training data and extract importances."""
    from prepare import load_problem, load_data, split_data, auto_preprocess
    from pipeline import PipelineConfig, build_sklearn_pipeline
    from search_space import get_registry

    X, y = load_data(problem)
    X = auto_preprocess(X)
    X_train, X_val, y_train, y_val = split_data(X, y)
    feature_names = list(X_train.columns)

    X_train_np = X_train.values if hasattr(X_train, "values") else X_train
    y_train_np = y_train.values if hasattr(y_train, "values") else y_train
    X_val_np = X_val.values if hasattr(X_val, "values") else X_val
    y_val_np = y_val.values if hasattr(y_val, "values") else y_val

    # Rebuild the best pipeline config from description
    desc = best_row["description"]
    parsed = parse_description(desc)
    config = rebuild_config_from_description(parsed["full_desc"])

    registry = get_registry()
    task_type = problem["task"]
    pipe = build_sklearn_pipeline(config, registry, task_type)

    print(f"  Retraining best pipeline for feature importances...", file=sys.stderr)
    pipe.fit(X_train_np, y_train_np)

    # Try to get feature importances
    algo = pipe.named_steps.get("algorithm")
    importances = None
    imp_type = "built-in"

    if hasattr(algo, "feature_importances_"):
        raw_imp = algo.feature_importances_
        # Map back to original feature names if feature selection was applied
        if "feature_selection" in pipe.named_steps:
            fs = pipe.named_steps["feature_selection"]
            if hasattr(fs, "get_support"):
                mask = fs.get_support()
                full_imp = np.zeros(len(feature_names))
                # The features reaching the algorithm went through all transforms
                # We need the count of features the algo actually saw
                if len(raw_imp) == mask.sum():
                    full_imp[mask] = raw_imp
                    importances = full_imp
        if importances is None:
            if len(raw_imp) == len(feature_names):
                importances = raw_imp
            else:
                # Feature count mismatch (PCA/PolynomialFeatures changed dims)
                # Fall back to permutation importance
                importances = None

    if importances is None:
        # Use permutation importance
        imp_type = "permutation"
        try:
            from sklearn.inspection import permutation_importance
            metric = problem["metric"]
            scoring_map = {
                "auc": "roc_auc", "accuracy": "accuracy", "f1": "f1_weighted",
                "logloss": "neg_log_loss", "rmse": "neg_root_mean_squared_error",
                "mse": "neg_mean_squared_error", "mae": "neg_mean_absolute_error",
                "r2": "r2",
            }
            scoring = scoring_map.get(metric, None)
            result = permutation_importance(pipe, X_val_np, y_val_np, n_repeats=10,
                                            random_state=42, scoring=scoring, n_jobs=-1)
            importances = result.importances_mean
            # These are aligned with the input features
            if len(importances) != len(feature_names):
                importances = importances[:len(feature_names)]
        except Exception as e:
            print(f"  Warning: permutation importance failed: {e}", file=sys.stderr)
            importances = np.ones(len(feature_names)) / len(feature_names)
            imp_type = "uniform (fallback)"

    return feature_names, importances, imp_type


def rebuild_config_from_description(desc_str):
    """Rebuild a PipelineConfig from a description string.

    This is a best-effort parser. It identifies known operator names and
    their parameters from the description format:
       StepName(param=val, ...) -> StepName2 -> Algorithm(param=val, ...)
    """
    from pipeline import PipelineConfig

    KNOWN_PREPARATORS = {
        "SimpleImputer_mean", "SimpleImputer_median", "SimpleImputer_most_frequent",
        "KNNImputer", "OutlierClipper", "Winsorizer",
    }
    KNOWN_PREPROCESSORS = {
        "StandardScaler", "MinMaxScaler", "RobustScaler", "PCA", "PolynomialFeatures",
    }
    KNOWN_SELECTORS = {"SelectKBest"}
    KNOWN_ALGORITHMS = {
        "RandomForest", "GradientBoosting", "ExtraTrees", "Ridge", "Lasso",
        "ElasticNet", "SVR", "KNeighbors", "DecisionTree", "AdaBoost",
        "XGBoost", "LightGBM", "MLP",
        "OLS", "Logistic", "SGD", "BayesianRidge", "GAM", "CoxPH",
    }

    parts = [p.strip() for p in desc_str.split(" -> ") if p.strip()]

    preparation = []
    preprocessing = []
    feature_selection = ("passthrough", {})
    algorithm = ("RandomForest", {})

    for part in parts:
        name, params = _parse_step(part)
        if name in KNOWN_PREPARATORS:
            preparation.append((name, params))
        elif name in KNOWN_PREPROCESSORS:
            preprocessing.append((name, params))
        elif name in KNOWN_SELECTORS:
            feature_selection = (name, params)
        elif name in KNOWN_ALGORITHMS:
            algorithm = (name, params)
        else:
            # Try to guess: if it's the last part, treat as algorithm
            if part == parts[-1]:
                algorithm = (name, params)
            else:
                preprocessing.append((name, params))

    return PipelineConfig(
        preparation=preparation,
        preprocessing=preprocessing,
        feature_selection=feature_selection,
        algorithm=algorithm,
    )


def _parse_step(step_str):
    """Parse 'Name(param1=val1, param2=val2)' into (name, {param: val})."""
    m = re.match(r"(\w+)\((.*)\)$", step_str, re.DOTALL)
    if not m:
        return step_str.strip(), {}
    name = m.group(1)
    params_str = m.group(2)

    params = {}
    if params_str.strip():
        # Handle nested tuples like hidden_sizes=(128, 64)
        # Use a simple state machine
        params = _parse_params(params_str)
    return name, params


def _parse_params(params_str):
    """Parse a comma-separated param string, handling nested parens."""
    params = {}
    depth = 0
    current = ""
    for ch in params_str:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            _add_param(params, current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        _add_param(params, current.strip())
    return params


def _add_param(params, kv_str):
    """Parse 'key=value' and add to params dict with type coercion."""
    if "=" not in kv_str:
        return
    key, val_str = kv_str.split("=", 1)
    key = key.strip()
    val_str = val_str.strip()
    params[key] = _coerce_value(val_str)


def _coerce_value(val_str):
    """Coerce a string value to the appropriate Python type."""
    if val_str == "None":
        return None
    if val_str == "True":
        return True
    if val_str == "False":
        return False

    # Tuple
    if val_str.startswith("(") and val_str.endswith(")"):
        inner = val_str[1:-1]
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        return tuple(_coerce_value(p) for p in parts)

    # Try int
    try:
        return int(val_str)
    except ValueError:
        pass

    # Try float
    try:
        return float(val_str)
    except ValueError:
        pass

    # String (strip quotes)
    if (val_str.startswith("'") and val_str.endswith("'")) or \
       (val_str.startswith('"') and val_str.endswith('"')):
        return val_str[1:-1]

    return val_str


def section_variable_importance(feature_names, importances, imp_type):
    """Print Section 2: Variable Importance as ASCII bar chart."""
    lines = []
    lines.append(C.colored("--- 2. Variable Importance ---", C.BOLD, C.YELLOW))
    lines.append(f"  Method: {imp_type}")
    lines.append("")

    # Sort by importance descending
    indices = np.argsort(importances)[::-1]
    max_imp = max(abs(importances)) if len(importances) > 0 else 1
    if max_imp == 0:
        max_imp = 1

    # Show top 20
    n_show = min(20, len(feature_names))
    max_name_len = max(len(feature_names[i]) for i in indices[:n_show])
    bar_width = 50

    for rank, idx in enumerate(indices[:n_show]):
        imp = importances[idx]
        name = feature_names[idx]
        bar_len = int(abs(imp) / max_imp * bar_width)
        bar = "\u2588" * bar_len
        color = C.GREEN if imp > 0 else C.RED
        lines.append(f"  {rank+1:3d}. {name:<{max_name_len}}  {color}{bar}{C.RESET} {imp:.4f}")

    if len(feature_names) > n_show:
        lines.append(f"  ... and {len(feature_names) - n_show} more features")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 2b: SHAP Analysis
# ---------------------------------------------------------------------------

def compute_shap_data(best_row, problem, max_shap_samples=500):
    """Compute SHAP values for the best pipeline. Returns (shap_values, X_sample, feature_names) or None."""
    import shap
    from prepare import load_data, split_data, auto_preprocess
    from pipeline import PipelineConfig, build_sklearn_pipeline
    from search_space import get_registry

    print(f"  Computing SHAP values (this may take a moment)...", file=sys.stderr)

    X, y = load_data(problem)
    X = auto_preprocess(X)
    X_train, X_val, y_train, y_val = split_data(X, y)
    feature_names = list(X_train.columns)

    X_train_np = X_train.values if hasattr(X_train, "values") else X_train
    y_train_np = y_train.values if hasattr(y_train, "values") else y_train
    X_val_np = X_val.values if hasattr(X_val, "values") else X_val

    desc = best_row["description"]
    parsed = parse_description(desc)
    config = rebuild_config_from_description(parsed["full_desc"])

    registry = get_registry()
    task_type = problem["task"]
    pipe = build_sklearn_pipeline(config, registry, task_type)
    pipe.fit(X_train_np, y_train_np)

    # Get the final estimator and the transformed data reaching it
    algo = pipe.named_steps.get("algorithm")

    # Transform validation data through all steps except the algorithm
    from sklearn.pipeline import Pipeline
    transform_steps = [(name, step) for name, step in pipe.named_steps.items() if name != "algorithm"]
    if transform_steps:
        transform_pipe = Pipeline(transform_steps)
        X_transformed = transform_pipe.transform(X_val_np)
    else:
        X_transformed = X_val_np

    # Get transformed feature names
    # If feature selection reduced dimensions, we need to map back
    n_transformed = X_transformed.shape[1]
    if n_transformed == len(feature_names):
        transformed_names = feature_names
    elif "feature_selection" in pipe.named_steps:
        fs = pipe.named_steps["feature_selection"]
        if hasattr(fs, "get_support"):
            mask = fs.get_support()
            transformed_names = [f for f, m in zip(feature_names, mask) if m]
        else:
            transformed_names = [f"feature_{i}" for i in range(n_transformed)]
    else:
        transformed_names = [f"feature_{i}" for i in range(n_transformed)]

    # Subsample for SHAP speed
    n_samples = min(max_shap_samples, X_transformed.shape[0])
    indices = np.random.RandomState(42).choice(X_transformed.shape[0], n_samples, replace=False)
    X_sample = X_transformed[indices]

    # Use TreeExplainer for tree models, KernelExplainer as fallback
    try:
        if hasattr(algo, "feature_importances_") or hasattr(algo, "get_booster"):
            explainer = shap.TreeExplainer(algo)
            shap_values = explainer.shap_values(X_sample)
            # For binary classification, shap_values may be a list [neg, pos]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            bg = shap.sample(X_transformed, min(100, X_transformed.shape[0]))
            if task_type == "classification" and hasattr(algo, "predict_proba"):
                explainer = shap.KernelExplainer(algo.predict_proba, bg)
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                explainer = shap.KernelExplainer(algo.predict, bg)
                shap_values = explainer.shap_values(X_sample)

        ev = explainer.expected_value
        return shap_values, X_sample, transformed_names, ev

    except Exception as e:
        print(f"  Warning: SHAP computation failed: {e}", file=sys.stderr)
        return None


def generate_shap_plots(shap_data, problem_name):
    """Generate SHAP waterfall and heatmap as base64 PNG strings."""
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    shap_values_arr, X_sample, feature_names, expected_value = shap_data

    # Handle expected_value for binary classification
    if isinstance(expected_value, (list, np.ndarray)):
        base_value = float(expected_value[1]) if len(expected_value) > 1 else float(expected_value[0])
    else:
        base_value = float(expected_value)

    # Create shap.Explanation object
    explanation = shap.Explanation(
        values=shap_values_arr,
        base_values=np.full(shap_values_arr.shape[0], base_value),
        data=X_sample,
        feature_names=feature_names,
    )

    plots = {}

    # --- Waterfall plot (single prediction - pick the median-prediction sample) ---
    try:
        pred_values = shap_values_arr.sum(axis=1)
        median_idx = np.argsort(pred_values)[len(pred_values) // 2]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        shap.plots.waterfall(explanation[median_idx], max_display=15, show=False)
        # Style the current figure for dark theme
        for a in fig.get_axes():
            a.set_facecolor("#0d1117")
            a.tick_params(colors="#c9d1d9")
            a.xaxis.label.set_color("#c9d1d9")
            a.yaxis.label.set_color("#c9d1d9")
            a.title.set_color("#c9d1d9")
            for spine in a.spines.values():
                spine.set_color("#30363d")
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        plots["waterfall"] = base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"  Warning: SHAP waterfall plot failed: {e}", file=sys.stderr)

    # --- Beeswarm / summary plot (heatmap-style) ---
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        shap.summary_plot(shap_values_arr, X_sample, feature_names=feature_names,
                          max_display=15, show=False, plot_size=None)
        for a in fig.get_axes():
            a.set_facecolor("#0d1117")
            a.tick_params(colors="#c9d1d9")
            a.xaxis.label.set_color("#c9d1d9")
            a.yaxis.label.set_color("#c9d1d9")
            a.title.set_color("#c9d1d9")
            for spine in a.spines.values():
                spine.set_color("#30363d")
        # Style colorbar if present
        for a in fig.get_axes():
            if hasattr(a, "collections") and len(a.collections) == 0:
                # This might be the colorbar axis
                a.tick_params(colors="#c9d1d9")
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        plots["beeswarm"] = base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"  Warning: SHAP beeswarm plot failed: {e}", file=sys.stderr)

    # --- Heatmap ---
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        shap.plots.heatmap(explanation, max_display=15, show=False)
        for a in fig.get_axes():
            a.set_facecolor("#0d1117")
            a.tick_params(colors="#c9d1d9")
            a.xaxis.label.set_color("#c9d1d9")
            a.yaxis.label.set_color("#c9d1d9")
            a.title.set_color("#c9d1d9")
            for spine in a.spines.values():
                spine.set_color("#30363d")
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        plots["heatmap"] = base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"  Warning: SHAP heatmap plot failed: {e}", file=sys.stderr)

    return plots


def section_shap_terminal(shap_data):
    """Print SHAP summary to terminal."""
    if shap_data is None:
        return C.colored("--- 2b. SHAP Analysis ---", C.BOLD, C.YELLOW) + "\n  Skipped (SHAP computation failed).\n\n"

    shap_values_arr, X_sample, feature_names, expected_value = shap_data

    lines = []
    lines.append(C.colored("--- 2b. SHAP Analysis ---", C.BOLD, C.YELLOW))
    lines.append(f"  Computed on {X_sample.shape[0]} samples, {len(feature_names)} features")
    lines.append("")

    # Mean absolute SHAP values (global importance)
    mean_abs_shap = np.mean(np.abs(shap_values_arr), axis=0)
    indices = np.argsort(mean_abs_shap)[::-1]
    max_shap = max(mean_abs_shap) if len(mean_abs_shap) > 0 else 1
    if max_shap == 0:
        max_shap = 1

    lines.append(f"  {C.BOLD}Mean |SHAP| (global feature impact):{C.RESET}")
    n_show = min(15, len(feature_names))
    max_name_len = max(len(feature_names[i]) for i in indices[:n_show])
    bar_width = 40

    for rank, idx in enumerate(indices[:n_show]):
        val = mean_abs_shap[idx]
        bar_len = int(val / max_shap * bar_width)
        bar = "\u2588" * bar_len
        lines.append(f"  {rank+1:3d}. {feature_names[idx]:<{max_name_len}}  {C.MAGENTA}{bar}{C.RESET} {val:.4f}")

    lines.append("")
    return "\n".join(lines)


def _html_shap_section(shap_plots):
    """Return HTML for the SHAP section with embedded plots."""
    if not shap_plots:
        return "<p style='color: #8b949e;'>SHAP analysis not available.</p>"

    html = ""

    if "waterfall" in shap_plots:
        html += """
        <h3>SHAP Waterfall (Single Prediction)</h3>
        <p style="color: #8b949e; margin-bottom: 10px;">
          Shows how each feature pushes the prediction from the base value (average model output)
          to the final prediction for a representative sample. Red bars push the prediction higher,
          blue bars push it lower.
        </p>
        """
        html += f'<img src="data:image/png;base64,{shap_plots["waterfall"]}" style="width:100%; max-width:800px; border-radius:6px; margin: 10px 0;" />'

    if "beeswarm" in shap_plots:
        html += """
        <h3 style="margin-top: 20px;">SHAP Beeswarm (Global Impact)</h3>
        <p style="color: #8b949e; margin-bottom: 10px;">
          Each dot is one sample. Position on x-axis shows the SHAP value (impact on prediction).
          Color indicates the feature value (red = high, blue = low). Features are sorted by overall
          importance. This reveals both the magnitude and direction of each feature's effect.
        </p>
        """
        html += f'<img src="data:image/png;base64,{shap_plots["beeswarm"]}" style="width:100%; max-width:800px; border-radius:6px; margin: 10px 0;" />'

    if "heatmap" in shap_plots:
        html += """
        <h3 style="margin-top: 20px;">SHAP Heatmap (Sample × Feature)</h3>
        <p style="color: #8b949e; margin-bottom: 10px;">
          Shows SHAP values across all samples (rows) and features (columns). Samples are ordered
          by similarity, revealing clusters of predictions driven by similar feature patterns.
          Red cells indicate positive SHAP contributions, blue cells indicate negative.
        </p>
        """
        html += f'<img src="data:image/png;base64,{shap_plots["heatmap"]}" style="width:100%; max-width:900px; border-radius:6px; margin: 10px 0;" />'

    return html


def _callout_shap():
    """Return interpretation callout for SHAP section."""
    return _html_callout(
        "Understanding SHAP Values",
        """<p><strong>SHAP (SHapley Additive exPlanations)</strong> provides a principled way to explain
        individual predictions by assigning each feature a contribution value based on cooperative game theory.</p>
        <ul style="margin: 8px 0; padding-left: 20px; color: #c9d1d9;">
          <li><strong>Waterfall plot:</strong> Decomposes a single prediction, showing how each feature
          shifts the output from the baseline (average prediction) to the final value. This answers
          "why did the model make this specific prediction?"</li>
          <li><strong>Beeswarm plot:</strong> Shows the distribution of SHAP values across all samples.
          Features at the top have the largest overall impact. The color-value relationship reveals
          direction — e.g., if high values of RECENT_3M_EVENTS (red dots) appear on the right (positive SHAP),
          that feature increases predicted propensity.</li>
          <li><strong>Heatmap:</strong> Reveals patterns across samples. Clusters of similar SHAP patterns
          indicate distinct subgroups in your data that the model treats differently.</li>
        </ul>
        <p><strong>Key difference from Variable Importance:</strong> Feature importance (Section 2) shows
        <em>how much</em> a feature matters overall. SHAP shows <em>how</em> and <em>in which direction</em>
        each feature drives predictions, including non-linear and interaction effects.</p>""",
        icon="\U0001f4a1"
    )


# ---------------------------------------------------------------------------
# Section 3: Feature Interactions
# ---------------------------------------------------------------------------

def compute_feature_interactions(best_row, problem, feature_names, importances, top_n=5):
    """Compute interaction strength for top feature pairs."""
    from prepare import load_data, split_data, auto_preprocess, evaluate
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    X, y = load_data(problem)
    X = auto_preprocess(X)
    X_train, X_val, y_train, y_val = split_data(X, y)

    X_train_np = X_train.values if hasattr(X_train, "values") else X_train
    y_train_np = y_train.values if hasattr(y_train, "values") else y_train
    X_val_np = X_val.values if hasattr(X_val, "values") else X_val
    y_val_np = y_val.values if hasattr(y_val, "values") else y_val

    metric = problem["metric"]
    task_type = problem["task"]

    # Use top 10 features
    top_indices = np.argsort(np.abs(importances))[::-1][:10]

    # Quick model for scoring individual and pair features
    if task_type == "classification":
        make_model = lambda: GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42, subsample=0.8)
    else:
        make_model = lambda: GradientBoostingRegressor(
            n_estimators=50, max_depth=3, random_state=42, subsample=0.8)

    def score_features(feat_indices):
        """Train a quick model on given feature indices and return score."""
        X_tr = X_train_np[:, feat_indices]
        X_va = X_val_np[:, feat_indices]
        model = make_model()
        try:
            model.fit(X_tr, y_train_np)
            if metric in ("auc", "logloss"):
                try:
                    y_pred = model.predict_proba(X_va)[:, 1]
                except (AttributeError, IndexError):
                    y_pred = model.predict(X_va)
            else:
                y_pred = model.predict(X_va)
            return evaluate(y_val_np, y_pred, metric, task_type)
        except Exception:
            return None

    print(f"  Computing feature interactions (this may take a moment)...", file=sys.stderr)

    # Cache individual scores
    individual_scores = {}
    for idx in top_indices:
        individual_scores[idx] = score_features([idx])

    # Compute pairwise interaction strength
    interactions = []
    n_top = len(top_indices)
    for i in range(n_top):
        for j in range(i + 1, n_top):
            fi, fj = top_indices[i], top_indices[j]
            pair_score = score_features([fi, fj])
            si = individual_scores.get(fi)
            sj = individual_scores.get(fj)

            if pair_score is not None and si is not None and sj is not None:
                # Interaction strength: how much the pair exceeds the better individual
                if problem["direction"] == "maximize":
                    expected = max(si, sj)
                    interaction = pair_score - expected
                else:
                    expected = min(si, sj)
                    interaction = expected - pair_score  # lower is better, so flip

                interactions.append((
                    feature_names[fi], feature_names[fj],
                    interaction, pair_score, si, sj
                ))

    # Sort by interaction strength descending
    interactions.sort(key=lambda x: x[2], reverse=True)
    return interactions[:top_n]


def section_feature_interactions(interactions, metric, direction):
    """Print Section 3: Feature Interactions."""
    lines = []
    lines.append(C.colored("--- 3. Feature Interactions ---", C.BOLD, C.YELLOW))
    lines.append(f"  Top feature pairs with synergistic effects ({metric}):")
    lines.append("")

    if not interactions:
        lines.append("  No interactions computed.")
        lines.append("")
        return "\n".join(lines)

    for rank, (f1, f2, strength, pair_sc, sc1, sc2) in enumerate(interactions):
        sign = "+" if strength > 0 else ""
        color = C.GREEN if strength > 0 else C.RED
        lines.append(f"  {rank+1}. {C.BOLD}{f1}{C.RESET} x {C.BOLD}{f2}{C.RESET}")
        lines.append(f"     Pair {metric}={pair_sc:.4f}  |  Individual: {sc1:.4f}, {sc2:.4f}  |  "
                      f"Interaction: {color}{sign}{strength:.4f}{C.RESET}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 4: Genetic Evolution Visualization
# ---------------------------------------------------------------------------

def compute_evolution_data(rows, direction):
    """Compute all evolution statistics from rows."""
    data = {
        "gen_best": {},          # gen -> best score so far
        "gen_scores": defaultdict(list),  # gen -> list of scores
        "gen_algorithms": defaultdict(lambda: Counter()),  # gen -> Counter of alg names
        "best_timeline": [],      # list of (gen, score, desc) when new best found
        "block_history": [],      # list of (gen, block_sig, score) for best pipeline trace
    }

    best_score = None
    best_desc = None

    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            gen = int(row["generation"])
            score = float(row["score"])
        except (ValueError, KeyError):
            continue

        desc = row.get("description", "")
        parsed = parse_description(desc)
        alg_name = extract_algorithm_name(parsed.get("algorithm", ""))

        data["gen_scores"][gen].append(score)
        data["gen_algorithms"][gen][alg_name] += 1

        is_new_best = False
        if best_score is None:
            is_new_best = True
        elif direction == "maximize" and score > best_score:
            is_new_best = True
        elif direction == "minimize" and score < best_score:
            is_new_best = True

        if is_new_best:
            best_score = score
            best_desc = desc
            data["best_timeline"].append((gen, score, desc))

        data["gen_best"][gen] = best_score

    return data


def section_evolution_block_diagram(evo_data, metric):
    """Print Section 4a: Block-switching diagram."""
    lines = []
    lines.append(C.colored("--- 4a. Best Pipeline Evolution (Block Switching) ---", C.BOLD, C.YELLOW))
    lines.append("")

    timeline = evo_data["best_timeline"]
    if not timeline:
        lines.append("  No improvements recorded.")
        lines.append("")
        return "\n".join(lines)

    prev_sig = None
    for gen, score, desc in timeline:
        parsed = parse_description(desc)
        sig = extract_block_signature(parsed)
        prep, preproc, fs, alg = sig

        # Determine what changed
        changes = []
        if prev_sig is not None:
            p_prep, p_preproc, p_fs, p_alg = prev_sig
            if prep != p_prep:
                changes.append("prep")
            if preproc != p_preproc:
                changes.append("preproc")
            if fs != p_fs:
                changes.append("feat_sel")
            if alg != p_alg:
                changes.append("algorithm")
            # If only params changed
            if not changes and parsed["full_desc"] != parse_description(evo_data["best_timeline"][
                    max(0, [t[0] for t in evo_data["best_timeline"]].index(gen) - 1)
                ][2])["full_desc"]:
                changes.append("params")

        change_str = ""
        if changes:
            change_str = C.colored(f"  << {'+'.join(changes)}", C.DIM, C.MAGENTA)

        # Format blocks
        prep_str = f"[{prep:^18}]" if prep != "-" else f"[{'---':^18}]"
        preproc_str = f"[{preproc:^18}]" if preproc != "-" else f"[{'---':^18}]"
        fs_str = f"[{fs:^18}]" if fs != "-" else f"[{'---':^18}]"

        # Truncate algorithm display
        alg_display = alg
        alg_step = parsed["steps"][-1] if parsed["steps"] else alg
        if len(alg_step) > 25:
            alg_display = alg_step[:22] + "..."
        else:
            alg_display = alg_step
        alg_str = f"[{alg_display:^25}]"

        gen_label = f"Gen {gen:>3}"
        score_str = f"{metric}={score:.4f}"

        lines.append(
            f"  {C.BOLD}{gen_label}{C.RESET}: "
            f"{C.CYAN}{prep_str}{C.RESET} -> "
            f"{C.CYAN}{preproc_str}{C.RESET} -> "
            f"{C.CYAN}{fs_str}{C.RESET} -> "
            f"{C.GREEN}{alg_str}{C.RESET}  "
            f"{C.BOLD}{score_str}{C.RESET}"
            f"{change_str}"
        )

        prev_sig = sig

    lines.append("")
    return "\n".join(lines)


def section_score_progression(evo_data, metric, direction):
    """Print Section 4b: ASCII sparkline of best score over generations."""
    lines = []
    lines.append(C.colored("--- 4b. Score Progression ---", C.BOLD, C.YELLOW))
    lines.append("")

    gen_best = evo_data["gen_best"]
    if not gen_best:
        lines.append("  No data.")
        lines.append("")
        return "\n".join(lines)

    max_gen = max(gen_best.keys())
    scores = []
    for g in range(max_gen + 1):
        if g in gen_best:
            scores.append(gen_best[g])
        elif scores:
            scores.append(scores[-1])
        else:
            scores.append(None)

    # Filter out Nones
    valid = [(i, s) for i, s in enumerate(scores) if s is not None]
    if not valid:
        lines.append("  No valid scores.")
        lines.append("")
        return "\n".join(lines)

    vals = [s for _, s in valid]
    min_s, max_s = min(vals), max(vals)
    spread = max_s - min_s if max_s != min_s else 1

    # ASCII chart: 60 cols wide, 15 rows tall
    chart_w = min(80, max_gen + 1)
    chart_h = 12
    sparkline_chars = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

    # Downsample if needed
    if max_gen + 1 > chart_w:
        step = (max_gen + 1) / chart_w
        sampled = []
        for i in range(chart_w):
            idx = int(i * step)
            if idx < len(scores) and scores[idx] is not None:
                sampled.append(scores[idx])
            elif sampled:
                sampled.append(sampled[-1])
        vals_sampled = sampled
    else:
        vals_sampled = [s for s in scores if s is not None]

    # Build sparkline
    spark = ""
    for v in vals_sampled:
        if direction == "minimize":
            # Invert for display: higher bar = better = lower score
            norm = 1.0 - (v - min_s) / spread if spread > 0 else 0.5
        else:
            norm = (v - min_s) / spread if spread > 0 else 0.5
        idx = int(norm * (len(sparkline_chars) - 1))
        idx = max(0, min(len(sparkline_chars) - 1, idx))
        spark += sparkline_chars[idx]

    lines.append(f"  {C.BOLD}Best {metric} over generations:{C.RESET}")
    lines.append(f"  {C.GREEN}{spark}{C.RESET}")
    lines.append(f"  {'Gen 0':<{len(vals_sampled)//2}}{'Gen ' + str(max_gen):>{len(vals_sampled) - len(vals_sampled)//2}}")
    lines.append(f"  Start: {vals[0]:.6f}  ->  End: {vals[-1]:.6f}")
    lines.append("")
    return "\n".join(lines)


def section_block_diversity(evo_data):
    """Print Section 4c: Algorithm diversity over time."""
    lines = []
    lines.append(C.colored("--- 4c. Algorithm Diversity Over Time ---", C.BOLD, C.YELLOW))
    lines.append("")

    gen_algs = evo_data["gen_algorithms"]
    if not gen_algs:
        lines.append("  No data.")
        lines.append("")
        return "\n".join(lines)

    max_gen = max(gen_algs.keys())

    # Collect all algorithm names
    all_algs = set()
    for counter in gen_algs.values():
        all_algs.update(counter.keys())
    all_algs = sorted(all_algs)

    # Sample generations for display (at most 12 time points)
    sample_gens = sorted(gen_algs.keys())
    if len(sample_gens) > 12:
        step = len(sample_gens) / 12
        sample_gens = [sample_gens[int(i * step)] for i in range(12)]

    # Header
    alg_width = max(len(a) for a in all_algs) if all_algs else 10
    header = f"  {'Gen':>5} | " + " | ".join(f"{a:>{max(4, len(a))}}" for a in all_algs) + " | total"
    lines.append(C.colored(header, C.DIM))
    lines.append("  " + "-" * (len(header) - 2))

    for gen in sample_gens:
        counter = gen_algs[gen]
        total = sum(counter.values())
        parts = []
        for alg in all_algs:
            count = counter.get(alg, 0)
            pct = count / total * 100 if total > 0 else 0
            w = max(4, len(alg))
            if pct > 30:
                parts.append(C.colored(f"{pct:>{w}.0f}%", C.GREEN))
            elif pct > 0:
                parts.append(f"{pct:>{w}.0f}%")
            else:
                parts.append(f"{'':>{w}} ")
        row_str = f"  {gen:>5} | " + " | ".join(parts) + f" | {total:>5}"
        lines.append(row_str)

    lines.append("")
    return "\n".join(lines)


def section_mutation_impact(evo_data):
    """Print Section 4d: Mutation/crossover impact analysis."""
    lines = []
    lines.append(C.colored("--- 4d. Genetic Operator Impact ---", C.BOLD, C.YELLOW))
    lines.append("")

    timeline = evo_data["best_timeline"]
    if len(timeline) < 2:
        lines.append("  Not enough improvements to analyze.")
        lines.append("")
        return "\n".join(lines)

    change_types = Counter()
    for i in range(1, len(timeline)):
        prev_desc = timeline[i - 1][2]
        curr_desc = timeline[i][2]

        prev_parsed = parse_description(prev_desc)
        curr_parsed = parse_description(curr_desc)
        prev_sig = extract_block_signature(prev_parsed)
        curr_sig = extract_block_signature(curr_parsed)

        if prev_sig[3] != curr_sig[3]:
            change_types["Algorithm swap"] += 1
        elif prev_sig[:3] != curr_sig[:3]:
            change_types["Block crossover/mutation"] += 1
        else:
            change_types["Hyperparameter tweak"] += 1

    total_improvements = len(timeline) - 1
    lines.append(f"  Total improvements found: {C.BOLD}{total_improvements}{C.RESET}")
    lines.append("")

    for change, count in change_types.most_common():
        pct = count / total_improvements * 100
        bar = "\u2588" * int(pct / 2)
        lines.append(f"  {change:<30} {count:>3} ({pct:5.1f}%) {C.CYAN}{bar}{C.RESET}")

    lines.append("")

    # Show improvement magnitudes
    lines.append(f"  {C.BOLD}Improvement trajectory:{C.RESET}")
    for i in range(1, min(len(timeline), 11)):
        prev_gen, prev_score, _ = timeline[i - 1]
        curr_gen, curr_score, curr_desc = timeline[i]
        delta = curr_score - prev_score
        parsed = parse_description(curr_desc)
        alg = extract_algorithm_name(parsed.get("algorithm", ""))
        sign = "+" if delta > 0 else ""
        color = C.GREEN if delta > 0 else C.RED
        lines.append(f"    Gen {prev_gen:>3} -> {curr_gen:>3}: {color}{sign}{delta:.6f}{C.RESET}  ({alg})")

    if len(timeline) > 11:
        lines.append(f"    ... and {len(timeline) - 11} more improvements")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML Report Generation
# ---------------------------------------------------------------------------

def _html_callout(title, body_html, icon="&#x1F4A1;"):
    """Return an HTML info-callout box with left border accent."""
    return f"""<div style="border-left: 4px solid #58a6ff; background: #0c2d6b22;
      border-radius: 0 6px 6px 0; padding: 14px 18px; margin: 14px 0;
      color: #a2d2fb; font-size: 13px; line-height: 1.7;">
  <div style="font-weight: 700; margin-bottom: 6px; color: #79c0ff; font-size: 14px;">{icon} {title}</div>
  {body_html}
</div>"""


def _callout_best_model(metric, score):
    """Interpretation callout for the Best Model Summary section."""
    if metric.lower() == "auc":
        if score >= 0.9:
            qual = "Excellent"
        elif score >= 0.8:
            qual = "Good"
        elif score >= 0.7:
            qual = "Fair"
        elif score >= 0.6:
            qual = "Poor"
        else:
            qual = "Near-random"
        ranges = (
            "<b>AUC interpretation:</b> 0.5 = random guessing, 1.0 = perfect separation.<br>"
            "0.9 &ndash; 1.0: Excellent &nbsp;|&nbsp; 0.8 &ndash; 0.9: Good &nbsp;|&nbsp; "
            "0.7 &ndash; 0.8: Fair &nbsp;|&nbsp; 0.6 &ndash; 0.7: Poor &nbsp;|&nbsp; &lt; 0.6: Near-random<br>"
            f"This model scores <b>{score:.4f}</b> &mdash; rated <b>{qual}</b>."
        )
    else:
        ranges = (
            f"The primary metric is <b>{_html_escape(metric)}</b>. "
            "Lower is better for error metrics (RMSE, MAE); higher is better for R&sup2;."
        )
    pipeline_note = (
        "<br><b>Pipeline steps explained:</b> "
        "<em>SimpleImputer</em> fills missing values; "
        "<em>Winsorizer / OutlierClipper</em> clips extreme values to reduce outlier influence; "
        "<em>StandardScaler / MinMaxScaler / RobustScaler</em> normalises feature ranges; "
        "<em>PCA</em> reduces dimensionality; "
        "<em>SelectKBest</em> keeps only the most predictive features; "
        "the final step is the ML algorithm (e.g., XGBoost, LightGBM, RandomForest)."
    )
    return _html_callout("How to read this section", ranges + pipeline_note)


def _callout_variable_importance():
    return _html_callout("Interpreting Variable Importance",
        "<b>Higher bar = more influential</b> on the model's predictions.<br>"
        "<b>Built-in importance</b> comes from the model itself (e.g., how often a feature is used in tree splits). "
        "<b>Permutation importance</b> measures how much the score degrades when a feature's values are randomly shuffled.<br>"
        "<em>Caveat:</em> Correlated features can split importance between them, making each look less important than it truly is.<br>"
        "<b>Business tip:</b> Focus data-quality and feature-engineering efforts on the top features. "
        "Features with near-zero importance may be candidates for removal, simplifying the model without hurting performance."
    )


def _callout_interactions():
    return _html_callout("Understanding Feature Interactions",
        "An <b>interaction</b> means two features <em>together</em> predict better than either alone.<br>"
        "The <b>interaction strength</b> number shows how much extra predictive power the pair provides "
        "beyond their individual contributions.<br>"
        "<b>How to use this:</b> Strong interactions suggest the relationship between these features and the target "
        "is non-linear or conditional &mdash; e.g., &ldquo;feature A only matters when feature B is high.&rdquo;<br>"
        "<em>Note on SHAP:</em> For deeper interaction analysis, "
        "SHAP (SHapley Additive exPlanations) values can decompose individual predictions. "
        "SHAP interaction values show exactly how pairs of features jointly influence each prediction, "
        "going beyond the aggregate view shown here."
    )


def _callout_block_switching():
    return _html_callout("Reading the Evolution Diagram",
        "This table uses a <b>genetic algorithm metaphor</b>: pipelines are &ldquo;organisms&rdquo; and their "
        "building blocks (imputer, scaler, feature selector, algorithm) are &ldquo;genes.&rdquo;<br>"
        "<b>Block switching</b> is like biological crossover &mdash; successful components from different parent "
        "pipelines are recombined to create better offspring.<br>"
        "Each row is a milestone where the best pipeline improved. Coloured tags on the right show "
        "<em>which type of genetic operation</em> caused the improvement (algorithm swap, block mutation, or parameter tweak)."
    )


def _callout_score_progression():
    return _html_callout("Interpreting the Score Curve",
        "<b>Steep early rise</b> = easy initial gains from finding a good algorithm family.<br>"
        "<b>Flat plateau</b> = diminishing returns; the search has likely found a near-optimal region.<br>"
        "If the curve plateaus <em>early</em>, the problem may be relatively simple or the budget generous. "
        "A <em>late</em> plateau suggests the search benefited from extended exploration.<br>"
        "The gap between start and end represents the <b>total improvement</b> found by evolution."
    )


def _callout_algorithm_diversity():
    return _html_callout("What Algorithm Diversity Shows",
        "<b>Algorithm diversity</b> measures how many different ML algorithm families are represented "
        "in the population at each generation.<br>"
        "Early generations typically explore many algorithm families. As evolution progresses, the "
        "best-performing type dominates. A rapid convergence to one algorithm (like XGBoost) suggests "
        "it is clearly superior for this problem.<br>"
        "<em>If diversity stays high throughout</em>, the problem may be ambiguous or the search space "
        "may need refinement."
    )


def _callout_operator_impact():
    return _html_callout("Genetic Operator Types Explained",
        "<b>Algorithm swap:</b> Completely replacing the ML algorithm (e.g., RandomForest &rarr; XGBoost). "
        "Highest impact, rarest.<br>"
        "<b>Block crossover / mutation:</b> Swapping or adding preprocessing / feature-selection steps. "
        "Moderate impact.<br>"
        "<b>Hyperparameter tweak:</b> Fine-tuning numbers within the same algorithm "
        "(e.g., learning_rate, n_estimators). Smallest per-step impact but most frequent.<br>"
        "<em>Typical pattern:</em> Algorithm swaps make the biggest early jumps, then block mutations "
        "and parameter tweaks refine. Diminishing improvements are normal and expected in evolutionary optimisation."
    )


def generate_html_report(problem, best_row, rows, evo_data, feature_names, importances,
                         imp_type, interactions, metric, direction, shap_plots=None):
    """Generate a complete single-problem HTML report with embedded SVG charts."""

    body_content = _generate_problem_body_html(
        problem, best_row, rows, evo_data, feature_names, importances,
        imp_type, interactions, metric, direction, shap_plots=shap_plots,
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoML Report: {problem['name']}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: #0d1117; color: #c9d1d9; line-height: 1.6; padding: 20px;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 12px; margin-bottom: 20px; font-size: 28px; }}
  h2 {{ color: #79c0ff; margin: 30px 0 15px 0; font-size: 20px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }}
  h3 {{ color: #d2a8ff; margin: 20px 0 10px 0; font-size: 16px; }}
  .card {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 20px; margin-bottom: 20px;
  }}
  .metric-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px; margin-bottom: 20px;
  }}
  .metric-box {{
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
    padding: 15px; text-align: center;
  }}
  .metric-box .label {{ color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }}
  .metric-box .value {{ color: #58a6ff; font-size: 24px; font-weight: bold; margin-top: 5px; }}
  .metric-box .value.highlight {{ color: #3fb950; font-size: 28px; }}
  .pipeline-step {{
    display: inline-block; background: #1f2937; border: 1px solid #374151;
    border-radius: 4px; padding: 4px 10px; margin: 2px; font-family: monospace; font-size: 13px;
    color: #7dd3fc;
  }}
  .arrow {{ color: #6b7280; margin: 0 4px; }}
  table {{
    width: 100%; border-collapse: collapse; margin: 10px 0;
    font-size: 13px;
  }}
  th {{ background: #21262d; color: #79c0ff; padding: 8px 12px; text-align: left; font-weight: 600; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: #1c2128; }}
  .bar-container {{ display: flex; align-items: center; gap: 8px; }}
  .bar {{ height: 18px; border-radius: 3px; min-width: 2px; }}
  .bar.positive {{ background: linear-gradient(90deg, #238636, #3fb950); }}
  .bar.negative {{ background: linear-gradient(90deg, #da3633, #f85149); }}
  .bar.neutral {{ background: linear-gradient(90deg, #1f6feb, #58a6ff); }}
  svg {{ width: 100%; height: auto; }}
  .tag {{
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 11px; font-weight: 600;
  }}
  .tag-green {{ background: #0d2818; color: #3fb950; border: 1px solid #238636; }}
  .tag-blue {{ background: #0c2d6b; color: #58a6ff; border: 1px solid #1f6feb; }}
  .tag-purple {{ background: #271052; color: #d2a8ff; border: 1px solid #8957e5; }}
  .tag-yellow {{ background: #2d2000; color: #d29922; border: 1px solid #9e6a03; }}
  .change-label {{ font-size: 11px; color: #d2a8ff; margin-left: 8px; }}
  .footer {{ text-align: center; color: #484f58; margin-top: 40px; padding: 20px; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
  <h1>AutoML Evolution Report: {problem['name']}</h1>
  {body_content}
  <div class="footer">
    Generated by AutoML Research Framework &bull; {time.strftime('%Y-%m-%d %H:%M:%S')}
  </div>
</div>
</body>
</html>"""

    return html


def _svg_score_progression(gen_best, metric, direction, max_gen, grad_id="scoreGrad"):
    """Generate SVG line chart of best score over generations."""
    if not gen_best:
        return "<p>No data available.</p>"

    w, h = 900, 280
    pad_l, pad_r, pad_t, pad_b = 70, 20, 20, 40

    chart_w = w - pad_l - pad_r
    chart_h = h - pad_t - pad_b

    gens_sorted = sorted(gen_best.keys())
    scores = [gen_best[g] for g in gens_sorted]
    min_s, max_s = min(scores), max(scores)
    spread = max_s - min_s if max_s != min_s else 1

    def x(gen):
        return pad_l + (gen - gens_sorted[0]) / max(1, gens_sorted[-1] - gens_sorted[0]) * chart_w

    def y(score):
        return pad_t + chart_h - (score - min_s) / spread * chart_h

    # Build path
    points = [(x(g), y(s)) for g, s in zip(gens_sorted, scores)]
    path_d = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in points)

    # Fill area
    fill_d = path_d + f" L {points[-1][0]:.1f},{pad_t + chart_h:.1f} L {points[0][0]:.1f},{pad_t + chart_h:.1f} Z"

    # Y-axis labels
    y_labels = ""
    n_ticks = 6
    for i in range(n_ticks):
        val = min_s + spread * i / (n_ticks - 1)
        yp = y(val)
        y_labels += f'<text x="{pad_l - 8}" y="{yp + 4}" text-anchor="end" fill="#8b949e" font-size="11">{val:.4f}</text>'
        y_labels += f'<line x1="{pad_l}" y1="{yp}" x2="{w - pad_r}" y2="{yp}" stroke="#21262d" stroke-width="1"/>'

    # X-axis labels
    x_labels = ""
    n_x = min(8, len(gens_sorted))
    step = max(1, len(gens_sorted) // n_x)
    for i in range(0, len(gens_sorted), step):
        g = gens_sorted[i]
        xp = x(g)
        x_labels += f'<text x="{xp}" y="{h - 8}" text-anchor="middle" fill="#8b949e" font-size="11">Gen {g}</text>'

    svg = f"""<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{w}" height="{h}" fill="#0d1117" rx="6"/>
  {y_labels}
  {x_labels}
  <path d="{fill_d}" fill="url(#{grad_id})" opacity="0.3"/>
  <path d="{path_d}" fill="none" stroke="#3fb950" stroke-width="2"/>
  <defs>
    <linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#3fb950" stop-opacity="0.4"/>
      <stop offset="100%" stop-color="#3fb950" stop-opacity="0.0"/>
    </linearGradient>
  </defs>
  <text x="{w/2}" y="{pad_t - 4}" text-anchor="middle" fill="#79c0ff" font-size="13" font-weight="bold">Best {metric} over generations</text>
</svg>"""
    return svg


def _svg_feature_importance(feature_names, importances, imp_type):
    """Generate SVG horizontal bar chart of feature importances."""
    indices = np.argsort(importances)[::-1]
    n_show = min(20, len(feature_names))
    max_imp = max(abs(importances)) if len(importances) > 0 else 1
    if max_imp == 0:
        max_imp = 1

    bar_h = 22
    gap = 4
    name_w = 180
    w = 900
    h = n_show * (bar_h + gap) + 30
    chart_w = w - name_w - 80

    bars = ""
    for rank, idx in enumerate(indices[:n_show]):
        imp = importances[idx]
        name = feature_names[idx]
        bar_len = abs(imp) / max_imp * chart_w
        yp = rank * (bar_h + gap) + 10
        color = "#3fb950" if imp >= 0 else "#f85149"

        bars += f'<text x="{name_w - 8}" y="{yp + bar_h * 0.7}" text-anchor="end" fill="#c9d1d9" font-size="12" font-family="monospace">{_html_escape(name)}</text>'
        bars += f'<rect x="{name_w}" y="{yp}" width="{max(2, bar_len):.1f}" height="{bar_h}" rx="3" fill="{color}" opacity="0.8"/>'
        bars += f'<text x="{name_w + bar_len + 6}" y="{yp + bar_h * 0.7}" fill="#8b949e" font-size="11">{imp:.4f}</text>'

    svg = f"""<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
  {bars}
</svg>"""
    return svg


def _svg_block_diversity(gen_algorithms):
    """Generate SVG stacked area chart of algorithm diversity."""
    if not gen_algorithms:
        return "<p>No data.</p>"

    all_algs = set()
    for counter in gen_algorithms.values():
        all_algs.update(counter.keys())
    all_algs = sorted(all_algs)

    gens_sorted = sorted(gen_algorithms.keys())
    if len(gens_sorted) < 2:
        return "<p>Not enough generations for diversity chart.</p>"

    # Colors for algorithms
    palette = [
        "#3fb950", "#58a6ff", "#d2a8ff", "#f0883e", "#da3633",
        "#d29922", "#3fb5e5", "#f778ba", "#79c0ff", "#7ee787",
        "#a5d6ff", "#ffd33d", "#ff7b72", "#d4a4f9",
    ]
    alg_colors = {alg: palette[i % len(palette)] for i, alg in enumerate(all_algs)}

    w, h = 900, 300
    pad_l, pad_r, pad_t, pad_b = 50, 150, 20, 40
    chart_w = w - pad_l - pad_r
    chart_h = h - pad_t - pad_b

    # Compute percentages
    pcts = []
    for g in gens_sorted:
        counter = gen_algorithms[g]
        total = sum(counter.values())
        row = {}
        for alg in all_algs:
            row[alg] = counter.get(alg, 0) / total if total > 0 else 0
        pcts.append(row)

    def x(i):
        return pad_l + i / max(1, len(gens_sorted) - 1) * chart_w

    def y(val):
        return pad_t + chart_h * (1 - val)

    # Build stacked areas (bottom to top)
    areas = ""
    for alg_idx, alg in enumerate(all_algs):
        points_top = []
        points_bottom = []
        for i in range(len(gens_sorted)):
            # Sum of all algorithms below this one
            bottom = sum(pcts[i].get(all_algs[j], 0) for j in range(alg_idx))
            top = bottom + pcts[i].get(alg, 0)
            xi = x(i)
            points_top.append((xi, y(top)))
            points_bottom.append((xi, y(bottom)))

        # Create closed path
        top_path = " L ".join(f"{px:.1f},{py:.1f}" for px, py in points_top)
        bottom_path = " L ".join(f"{px:.1f},{py:.1f}" for px, py in reversed(points_bottom))
        path = f"M {top_path} L {bottom_path} Z"
        color = alg_colors.get(alg, "#888")
        areas += f'<path d="{path}" fill="{color}" opacity="0.7"/>'

    # X axis labels
    x_labels = ""
    n_x = min(8, len(gens_sorted))
    step = max(1, len(gens_sorted) // n_x)
    for i in range(0, len(gens_sorted), step):
        g = gens_sorted[i]
        xp = x(i)
        x_labels += f'<text x="{xp}" y="{h - 8}" text-anchor="middle" fill="#8b949e" font-size="11">Gen {g}</text>'

    # Legend
    legend = ""
    for i, alg in enumerate(all_algs):
        ly = pad_t + 15 + i * 20
        lx = w - pad_r + 15
        color = alg_colors.get(alg, "#888")
        legend += f'<rect x="{lx}" y="{ly}" width="12" height="12" rx="2" fill="{color}" opacity="0.8"/>'
        legend += f'<text x="{lx + 18}" y="{ly + 10}" fill="#c9d1d9" font-size="11">{_html_escape(alg)}</text>'

    # Y axis
    y_labels = ""
    for pct_val in [0, 25, 50, 75, 100]:
        yp = y(pct_val / 100)
        y_labels += f'<text x="{pad_l - 8}" y="{yp + 4}" text-anchor="end" fill="#8b949e" font-size="11">{pct_val}%</text>'
        y_labels += f'<line x1="{pad_l}" y1="{yp}" x2="{pad_l + chart_w}" y2="{yp}" stroke="#21262d" stroke-width="1"/>'

    svg = f"""<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{w}" height="{h}" fill="#0d1117" rx="6"/>
  {y_labels}
  {areas}
  {x_labels}
  {legend}
  <text x="{pad_l + chart_w/2}" y="{pad_t - 4}" text-anchor="middle" fill="#79c0ff" font-size="13" font-weight="bold">Population Algorithm Distribution</text>
</svg>"""
    return svg


def _html_pipeline_evolution(timeline, metric):
    """Generate HTML table showing pipeline block evolution."""
    if not timeline:
        return "<p>No improvements recorded.</p>"

    rows_html = ""
    prev_sig = None
    for gen, score, desc in timeline:
        parsed = parse_description(desc)
        sig = extract_block_signature(parsed)
        prep, preproc, fs, alg = sig

        changes = []
        if prev_sig is not None:
            p_prep, p_preproc, p_fs, p_alg = prev_sig
            if prep != p_prep:
                changes.append("prep")
            if preproc != p_preproc:
                changes.append("preproc")
            if fs != p_fs:
                changes.append("feat_sel")
            if alg != p_alg:
                changes.append("algorithm")
            if not changes:
                changes.append("params")

        change_tags = ""
        tag_map = {
            "prep": "tag-blue", "preproc": "tag-blue",
            "feat_sel": "tag-purple", "algorithm": "tag-yellow",
            "params": "tag-green",
        }
        for ch in changes:
            cls = tag_map.get(ch, "tag-green")
            change_tags += f'<span class="tag {cls}">{ch}</span> '

        def block_cell(val):
            if val == "-":
                return '<td style="color: #484f58; text-align: center;">--</td>'
            return f'<td><span class="pipeline-step">{_html_escape(val)}</span></td>'

        rows_html += f"""<tr>
  <td style="font-weight: bold;">{gen}</td>
  {block_cell(prep)}
  {block_cell(preproc)}
  {block_cell(fs)}
  <td><span class="pipeline-step" style="color: #3fb950;">{_html_escape(alg)}</span></td>
  <td style="font-weight: bold; color: #3fb950;">{score:.6f}</td>
  <td>{change_tags}</td>
</tr>"""
        prev_sig = sig

    return f"""<table>
<thead><tr>
  <th>Gen</th><th>Preparation</th><th>Preprocessing</th><th>Feature Sel.</th><th>Algorithm</th><th>{metric}</th><th>Changed</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>"""


def _html_interactions(interactions, metric):
    """Generate HTML table for feature interactions."""
    if not interactions:
        return "<p>No interactions computed.</p>"

    rows_html = ""
    for rank, (f1, f2, strength, pair_sc, sc1, sc2) in enumerate(interactions):
        color = "#3fb950" if strength > 0 else "#f85149"
        sign = "+" if strength > 0 else ""
        bar_w = min(200, abs(strength) / max(abs(interactions[0][2]), 0.001) * 200)
        bar_class = "positive" if strength > 0 else "negative"

        rows_html += f"""<tr>
  <td>{rank + 1}</td>
  <td><b>{_html_escape(f1)}</b></td>
  <td><b>{_html_escape(f2)}</b></td>
  <td>{pair_sc:.4f}</td>
  <td>{sc1:.4f}</td>
  <td>{sc2:.4f}</td>
  <td>
    <div class="bar-container">
      <div class="bar {bar_class}" style="width: {max(4, bar_w):.0f}px;"></div>
      <span style="color: {color};">{sign}{strength:.4f}</span>
    </div>
  </td>
</tr>"""

    return f"""<table>
<thead><tr>
  <th>#</th><th>Feature A</th><th>Feature B</th><th>Pair {metric}</th>
  <th>A alone</th><th>B alone</th><th>Interaction</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>"""


def _html_operator_impact(timeline):
    """Generate HTML for operator impact analysis."""
    if len(timeline) < 2:
        return "<p>Not enough improvements to analyze.</p>"

    change_types = Counter()
    for i in range(1, len(timeline)):
        prev_desc = timeline[i - 1][2]
        curr_desc = timeline[i][2]
        prev_sig = extract_block_signature(parse_description(prev_desc))
        curr_sig = extract_block_signature(parse_description(curr_desc))

        if prev_sig[3] != curr_sig[3]:
            change_types["Algorithm swap"] += 1
        elif prev_sig[:3] != curr_sig[:3]:
            change_types["Block crossover/mutation"] += 1
        else:
            change_types["Hyperparameter tweak"] += 1

    total = len(timeline) - 1
    bars_html = ""
    for change, count in change_types.most_common():
        pct = count / total * 100
        bar_w = pct / 100 * 400
        bars_html += f"""<div style="display: flex; align-items: center; margin: 8px 0; gap: 12px;">
  <div style="width: 200px; text-align: right; color: #c9d1d9;">{change}</div>
  <div class="bar neutral" style="width: {max(4, bar_w):.0f}px;"></div>
  <span style="color: #8b949e;">{count} ({pct:.0f}%)</span>
</div>"""

    # Improvement trajectory
    traj_rows = ""
    for i in range(1, min(len(timeline), 16)):
        prev_gen, prev_score, _ = timeline[i - 1]
        curr_gen, curr_score, curr_desc = timeline[i]
        delta = curr_score - prev_score
        parsed = parse_description(curr_desc)
        alg = extract_algorithm_name(parsed.get("algorithm", ""))
        sign = "+" if delta > 0 else ""
        color = "#3fb950" if delta > 0 else "#f85149"
        traj_rows += f"""<tr>
  <td>Gen {prev_gen} &rarr; {curr_gen}</td>
  <td style="color: {color}; font-weight: bold;">{sign}{delta:.6f}</td>
  <td>{_html_escape(alg)}</td>
</tr>"""

    return f"""<h3>Operator Contribution to Improvements</h3>
<p style="color: #8b949e; margin-bottom: 10px;">Total improvements: {total}</p>
{bars_html}
<h3 style="margin-top: 20px;">Improvement Trajectory</h3>
<table><thead><tr><th>Transition</th><th>Delta</th><th>Algorithm</th></tr></thead>
<tbody>{traj_rows}</tbody></table>"""


def _html_escape(text):
    """Simple HTML escaping."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------------------
# Generate the inner HTML content for one problem (no <html>/<head> wrapper).
# Reused by both single-problem and combined-report paths.
# ---------------------------------------------------------------------------

def _generate_problem_body_html(problem, best_row, rows, evo_data, feature_names,
                                importances, imp_type, interactions, metric, direction,
                                shap_plots=None):
    """Return the inner HTML content for one problem (sections 1-4d)."""
    score = float(best_row["score"])
    parsed = parse_description(best_row["description"])
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    gens = set()
    for r in rows:
        try:
            gens.add(int(r["generation"]))
        except (ValueError, KeyError):
            pass
    max_gen = max(gens) if gens else 0

    # Unique SVG gradient IDs per problem to avoid collisions in combined report
    problem_slug = problem["name"].replace("-", "_").replace(" ", "_")

    score_svg = _svg_score_progression(evo_data["gen_best"], metric, direction, max_gen,
                                       grad_id=f"scoreGrad_{problem_slug}")
    imp_svg = _svg_feature_importance(feature_names, importances, imp_type)
    diversity_svg = _svg_block_diversity(evo_data["gen_algorithms"])
    pipeline_table = _html_pipeline_evolution(evo_data["best_timeline"], metric)
    interactions_table = _html_interactions(interactions, metric)
    operator_html = _html_operator_impact(evo_data["best_timeline"])

    return f"""
  <!-- Section 1: Summary -->
  <h2>1. Best Model Summary</h2>
  {_callout_best_model(metric, score)}
  <div class="metric-grid">
    <div class="metric-box">
      <div class="label">Best {metric}</div>
      <div class="value highlight">{score:.6f}</div>
    </div>
    <div class="metric-box">
      <div class="label">Generations</div>
      <div class="value">{max_gen}</div>
    </div>
    <div class="metric-box">
      <div class="label">Evaluations</div>
      <div class="value">{len(rows)}</div>
    </div>
    <div class="metric-box">
      <div class="label">Success Rate</div>
      <div class="value">{len(ok_rows)/max(len(rows),1)*100:.0f}%</div>
    </div>
    <div class="metric-box">
      <div class="label">Found at Gen</div>
      <div class="value">{best_row['generation']}</div>
    </div>
    <div class="metric-box">
      <div class="label">Training Time</div>
      <div class="value">{best_row.get('elapsed_s','?')}s</div>
    </div>
  </div>

  <div class="card">
    <h3>Winning Pipeline</h3>
    <p style="margin-top: 10px;">
      {'<span class="arrow"> &rarr; </span>'.join(f'<span class="pipeline-step">{step}</span>' for step in parsed["steps"])}
    </p>
  </div>

  <!-- Section 2: Feature Importance -->
  <h2>2. Variable Importance</h2>
  {_callout_variable_importance()}
  <div class="card">
    <p style="color: #8b949e; margin-bottom: 10px;">Method: {imp_type}</p>
    {imp_svg}
  </div>

  <!-- Section 2b: SHAP Analysis -->
  <h2>2b. SHAP Analysis</h2>
  {_callout_shap()}
  <div class="card">
    {_html_shap_section(shap_plots) if shap_plots else '<p style="color: #8b949e;">SHAP analysis not available for this run.</p>'}
  </div>

  <!-- Section 3: Feature Interactions -->
  <h2>3. Feature Interactions</h2>
  {_callout_interactions()}
  <div class="card">
    {interactions_table}
  </div>

  <!-- Section 4a: Pipeline Evolution -->
  <h2>4a. Best Pipeline Evolution</h2>
  {_callout_block_switching()}
  <div class="card">
    {pipeline_table}
  </div>

  <!-- Section 4b: Score Progression -->
  <h2>4b. Score Progression</h2>
  {_callout_score_progression()}
  <div class="card">
    {score_svg}
  </div>

  <!-- Section 4c: Algorithm Diversity -->
  <h2>4c. Algorithm Diversity Over Time</h2>
  {_callout_algorithm_diversity()}
  <div class="card">
    {diversity_svg}
  </div>

  <!-- Section 4d: Operator Impact -->
  <h2>4d. Genetic Operator Impact</h2>
  {_callout_operator_impact()}
  <div class="card">
    {operator_html}
  </div>
"""


# ---------------------------------------------------------------------------
# Combined tabbed HTML report
# ---------------------------------------------------------------------------

def _pretty_problem_name(name):
    """Convert 'email-propensity' to 'Email Propensity'."""
    return name.replace("-", " ").replace("_", " ").title()


def _svg_combined_score_progression(all_problem_data):
    """SVG showing score progression curves for all problems on one chart.

    all_problem_data: list of (problem_name, gen_best_dict, metric, direction, color)
    """
    if not all_problem_data:
        return "<p>No data available.</p>"

    w, h = 900, 320
    pad_l, pad_r, pad_t, pad_b = 70, 170, 30, 45
    chart_w = w - pad_l - pad_r
    chart_h = h - pad_t - pad_b

    # Collect global bounds
    all_scores = []
    all_gens = []
    for _, gen_best, _, _, _ in all_problem_data:
        if gen_best:
            all_scores.extend(gen_best.values())
            all_gens.extend(gen_best.keys())

    if not all_scores:
        return "<p>No score data available.</p>"

    min_s = min(all_scores)
    max_s = max(all_scores)
    spread = max_s - min_s if max_s != min_s else 1
    min_g = min(all_gens)
    max_g = max(all_gens)
    g_range = max(1, max_g - min_g)

    def x(gen):
        return pad_l + (gen - min_g) / g_range * chart_w

    def y(score):
        return pad_t + chart_h - (score - min_s) / spread * chart_h

    # Y-axis
    y_labels = ""
    for i in range(6):
        val = min_s + spread * i / 5
        yp = y(val)
        y_labels += f'<text x="{pad_l - 8}" y="{yp + 4}" text-anchor="end" fill="#8b949e" font-size="11">{val:.4f}</text>'
        y_labels += f'<line x1="{pad_l}" y1="{yp}" x2="{pad_l + chart_w}" y2="{yp}" stroke="#21262d" stroke-width="1"/>'

    # X-axis
    x_labels = ""
    for gv in range(min_g, max_g + 1, max(1, g_range // 7)):
        xp = x(gv)
        x_labels += f'<text x="{xp}" y="{h - 8}" text-anchor="middle" fill="#8b949e" font-size="11">Gen {gv}</text>'

    # Lines and legend
    paths = ""
    legend = ""
    for idx, (pname, gen_best, metric, direction, color) in enumerate(all_problem_data):
        gens_sorted = sorted(gen_best.keys())
        if not gens_sorted:
            continue
        scores = [gen_best[g] for g in gens_sorted]
        points = [(x(g), y(s)) for g, s in zip(gens_sorted, scores)]
        path_d = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        paths += f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.5" opacity="0.9"/>'
        # Legend
        ly = pad_t + 10 + idx * 22
        lx = pad_l + chart_w + 20
        legend += f'<rect x="{lx}" y="{ly}" width="14" height="14" rx="3" fill="{color}" opacity="0.9"/>'
        legend += f'<text x="{lx + 20}" y="{ly + 11}" fill="#c9d1d9" font-size="12">{_html_escape(_pretty_problem_name(pname))}</text>'

    svg = f"""<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{w}" height="{h}" fill="#0d1117" rx="6"/>
  {y_labels}{x_labels}{paths}{legend}
  <text x="{pad_l + chart_w/2}" y="{pad_t - 10}" text-anchor="middle" fill="#79c0ff" font-size="14" font-weight="bold">Score Progression Comparison</text>
</svg>"""
    return svg


def _generate_summary_tab(problem_results):
    """Generate the content for the Summary comparison tab.

    problem_results: list of dicts with keys:
      name, metric, direction, score, max_gen, n_evals, best_alg,
      convergence_gen, evo_data, color
    """
    # Comparison table
    table_rows = ""
    for pr in problem_results:
        table_rows += f"""<tr>
  <td style="font-weight:bold;">{_html_escape(_pretty_problem_name(pr['name']))}</td>
  <td style="color:#3fb950; font-weight:bold;">{pr['score']:.6f}</td>
  <td>{pr['n_evals']}</td>
  <td>{pr['max_gen']}</td>
  <td><span class="pipeline-step" style="color:#3fb950;">{_html_escape(pr['best_alg'])}</span></td>
  <td>{pr['convergence_gen']}</td>
</tr>"""

    comparison_table = f"""<table>
<thead><tr>
  <th>Problem</th><th>Best Score</th><th>Evaluations</th><th>Generations</th><th>Best Algorithm</th><th>Convergence Gen</th>
</tr></thead>
<tbody>{table_rows}</tbody>
</table>"""

    # Combined score chart
    chart_data = []
    problem_colors = ["#3fb950", "#58a6ff", "#d2a8ff", "#f0883e", "#da3633", "#d29922"]
    for i, pr in enumerate(problem_results):
        color = problem_colors[i % len(problem_colors)]
        pr["color"] = color
        chart_data.append((
            pr["name"],
            pr["evo_data"]["gen_best"],
            pr["metric"],
            pr["direction"],
            color,
        ))
    combined_svg = _svg_combined_score_progression(chart_data)

    # Key findings
    if problem_results:
        best_pr = max(problem_results, key=lambda p: p["score"])
        worst_pr = min(problem_results, key=lambda p: p["score"])
        alg_counts = Counter(pr["best_alg"] for pr in problem_results)
        common_alg = alg_counts.most_common(1)[0][0] if alg_counts else "N/A"

        findings = (
            f"<b>Highest score:</b> {_pretty_problem_name(best_pr['name'])} "
            f"({best_pr['score']:.6f})<br>"
            f"<b>Lowest score:</b> {_pretty_problem_name(worst_pr['name'])} "
            f"({worst_pr['score']:.6f})<br>"
            f"<b>Most common winning algorithm:</b> {common_alg}<br>"
        )
        # Check for web propensity gap
        web_pr = next((p for p in problem_results if "web" in p["name"].lower()), None)
        if web_pr:
            gap = best_pr["score"] - web_pr["score"] if best_pr["name"] != web_pr["name"] else 0
            if gap > 0.001:
                findings += (
                    f"<br><b>Web Propensity Gap:</b> Web propensity scores "
                    f"{gap:.4f} points below the best problem. "
                    "This may indicate that web engagement signals are noisier or that "
                    "the feature set needs enrichment for web-channel prediction."
                )
        findings_html = _html_callout("Key Findings", findings, icon="&#x1F50D;")
    else:
        findings_html = ""

    return f"""
  <h2>Cross-Problem Comparison</h2>
  {findings_html}
  <div class="card">
    {comparison_table}
  </div>

  <h2>Combined Score Progression</h2>
  <div class="card">
    {combined_svg}
  </div>
"""


def generate_combined_html_report(problem_bodies, problem_results):
    """Generate a single HTML file with tabs for multiple problems + a Summary tab.

    problem_bodies: list of (display_name, body_html) for each problem tab
    problem_results: list of dicts for the summary tab (see _generate_summary_tab)
    """
    # Build tab bar + tab panels
    n_tabs = len(problem_bodies) + 1  # +1 for Summary
    tab_ids = ["tab-summary"] + [f"tab-{i}" for i in range(len(problem_bodies))]
    tab_labels = ["Summary"] + [name for name, _ in problem_bodies]

    # CSS for tabs
    tab_css = """
    .tab-bar {
      display: flex; gap: 0; border-bottom: 2px solid #30363d;
      margin-bottom: 24px; flex-wrap: wrap;
    }
    .tab-bar input[type="radio"] { display: none; }
    .tab-bar label {
      padding: 10px 22px; cursor: pointer; font-size: 14px; font-weight: 600;
      color: #8b949e; border-bottom: 3px solid transparent;
      transition: color 0.2s, border-color 0.2s, background 0.2s;
      margin-bottom: -2px; border-radius: 6px 6px 0 0;
    }
    .tab-bar label:hover { color: #c9d1d9; background: #161b2244; }
    .tab-panel {
      display: none; animation: tabFadeIn 0.3s ease;
    }
    @keyframes tabFadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    """
    # Each radio:checked shows its panel, highlights its label
    for i, tid in enumerate(tab_ids):
        tab_css += f"""
    #{tid}:checked ~ .tab-bar label[for="{tid}"] {{
      color: #58a6ff; border-bottom-color: #58a6ff; background: #161b22;
    }}
    #{tid}:checked ~ .tab-panel-{tid} {{ display: block; }}
    """

    # Build radio inputs (outside .tab-bar so CSS sibling selectors work)
    radios = ""
    for i, tid in enumerate(tab_ids):
        checked = ' checked="checked"' if i == 0 else ""
        radios += f'<input type="radio" name="tabs" id="{tid}"{checked}>\n'

    # Build label bar
    labels = '<div class="tab-bar">\n'
    for tid, label in zip(tab_ids, tab_labels):
        labels += f'  <label for="{tid}">{_html_escape(label)}</label>\n'
    labels += '</div>\n'

    # Build panels
    summary_body = _generate_summary_tab(problem_results)
    panels = f'<div class="tab-panel tab-panel-tab-summary">\n{summary_body}\n</div>\n'
    for i, (name, body) in enumerate(problem_bodies):
        tid = f"tab-{i}"
        panels += f'<div class="tab-panel tab-panel-{tid}">\n{body}\n</div>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoML Combined Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: #0d1117; color: #c9d1d9; line-height: 1.6; padding: 20px;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 12px; margin-bottom: 20px; font-size: 28px; }}
  h2 {{ color: #79c0ff; margin: 30px 0 15px 0; font-size: 20px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }}
  h3 {{ color: #d2a8ff; margin: 20px 0 10px 0; font-size: 16px; }}
  .card {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 20px; margin-bottom: 20px;
  }}
  .metric-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px; margin-bottom: 20px;
  }}
  .metric-box {{
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
    padding: 15px; text-align: center;
  }}
  .metric-box .label {{ color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }}
  .metric-box .value {{ color: #58a6ff; font-size: 24px; font-weight: bold; margin-top: 5px; }}
  .metric-box .value.highlight {{ color: #3fb950; font-size: 28px; }}
  .pipeline-step {{
    display: inline-block; background: #1f2937; border: 1px solid #374151;
    border-radius: 4px; padding: 4px 10px; margin: 2px; font-family: monospace; font-size: 13px;
    color: #7dd3fc;
  }}
  .arrow {{ color: #6b7280; margin: 0 4px; }}
  table {{
    width: 100%; border-collapse: collapse; margin: 10px 0;
    font-size: 13px;
  }}
  th {{ background: #21262d; color: #79c0ff; padding: 8px 12px; text-align: left; font-weight: 600; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: #1c2128; }}
  .bar-container {{ display: flex; align-items: center; gap: 8px; }}
  .bar {{ height: 18px; border-radius: 3px; min-width: 2px; }}
  .bar.positive {{ background: linear-gradient(90deg, #238636, #3fb950); }}
  .bar.negative {{ background: linear-gradient(90deg, #da3633, #f85149); }}
  .bar.neutral {{ background: linear-gradient(90deg, #1f6feb, #58a6ff); }}
  svg {{ width: 100%; height: auto; }}
  .tag {{
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 11px; font-weight: 600;
  }}
  .tag-green {{ background: #0d2818; color: #3fb950; border: 1px solid #238636; }}
  .tag-blue {{ background: #0c2d6b; color: #58a6ff; border: 1px solid #1f6feb; }}
  .tag-purple {{ background: #271052; color: #d2a8ff; border: 1px solid #8957e5; }}
  .tag-yellow {{ background: #2d2000; color: #d29922; border: 1px solid #9e6a03; }}
  .change-label {{ font-size: 11px; color: #d2a8ff; margin-left: 8px; }}
  .footer {{ text-align: center; color: #484f58; margin-top: 40px; padding: 20px; font-size: 12px; }}
  {tab_css}
</style>
</head>
<body>
<div class="container">
  <h1>AutoML Evolution Report &mdash; Combined</h1>
  {radios}
  {labels}
  {panels}
  <div class="footer">
    Generated by AutoML Research Framework &bull; {time.strftime('%Y-%m-%d %H:%M:%S')}
  </div>
</div>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Per-problem data collection (used by both single and multi paths)
# ---------------------------------------------------------------------------

def _collect_problem_data(problem_path, all_rows, runs):
    """Run the full analysis for one problem and return all data needed for reports.

    Returns a dict with: problem, best_row, rows, evo_data, feature_names,
    importances, imp_type, interactions, metric, direction, terminal_output
    """
    from prepare import load_problem

    problem = load_problem(problem_path)
    metric = problem["metric"]
    direction = problem["direction"]
    problem_name = problem["name"]

    start, end = pick_run_for_problem(all_rows, runs, problem)
    rows = all_rows[start:end]
    run_idx = next((i for i, (s, e) in enumerate(runs) if s == start), len(runs))
    print(f"  [{problem_name}] Using run {run_idx + 1} ({len(rows)} rows, rows {start}-{end})",
          file=sys.stderr)

    best_row = find_best_row(rows, direction)
    if not best_row:
        print(f"  [{problem_name}] WARNING: No successful pipelines found.", file=sys.stderr)
        return None

    # Terminal output
    output = section_best_model(rows, problem, best_row)

    # Feature importance
    try:
        feature_names, importances, imp_type = compute_feature_importances(best_row, problem)
        output += section_variable_importance(feature_names, importances, imp_type)
    except Exception as e:
        feature_names, importances, imp_type = [], np.array([]), "failed"
        output += C.colored("--- 2. Variable Importance ---", C.BOLD, C.YELLOW) + "\n"
        output += f"  {C.RED}Failed to compute: {e}{C.RESET}\n\n"

    # Feature interactions
    interactions = []
    try:
        if len(feature_names) > 1 and len(importances) > 1:
            interactions = compute_feature_interactions(
                best_row, problem, feature_names, importances, top_n=5)
            output += section_feature_interactions(interactions, metric, direction)
        else:
            output += C.colored("--- 3. Feature Interactions ---", C.BOLD, C.YELLOW) + "\n"
            output += "  Skipped (insufficient features).\n\n"
    except Exception as e:
        output += C.colored("--- 3. Feature Interactions ---", C.BOLD, C.YELLOW) + "\n"
        output += f"  {C.RED}Failed to compute: {e}{C.RESET}\n\n"

    # SHAP analysis
    shap_data = None
    shap_plots = None
    try:
        shap_data = compute_shap_data(best_row, problem)
        if shap_data is not None:
            output += section_shap_terminal(shap_data)
            shap_plots = generate_shap_plots(shap_data, problem_name)
    except Exception as e:
        output += C.colored("--- 2b. SHAP Analysis ---", C.BOLD, C.YELLOW) + "\n"
        output += f"  {C.RED}Failed to compute: {e}{C.RESET}\n\n"

    # Evolution
    evo_data = compute_evolution_data(rows, direction)
    output += section_evolution_block_diagram(evo_data, metric)
    output += section_score_progression(evo_data, metric, direction)
    output += section_block_diversity(evo_data)
    output += section_mutation_impact(evo_data)

    # Derived stats
    gens = set()
    for r in rows:
        try:
            gens.add(int(r["generation"]))
        except (ValueError, KeyError):
            pass
    max_gen = max(gens) if gens else 0

    parsed = parse_description(best_row["description"])
    best_alg = extract_algorithm_name(parsed.get("algorithm", ""))

    # Convergence gen: the generation of the last improvement
    timeline = evo_data["best_timeline"]
    convergence_gen = timeline[-1][0] if timeline else 0

    return {
        "problem": problem,
        "best_row": best_row,
        "rows": rows,
        "evo_data": evo_data,
        "feature_names": feature_names,
        "importances": importances,
        "imp_type": imp_type,
        "interactions": interactions,
        "metric": metric,
        "direction": direction,
        "terminal_output": output,
        "score": float(best_row["score"]),
        "max_gen": max_gen,
        "n_evals": len(rows),
        "best_alg": best_alg,
        "convergence_gen": convergence_gen,
        "name": problem_name,
        "shap_plots": shap_plots,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from prepare import load_problem

    # Determine problem paths
    args = [a for a in sys.argv[1:] if not a.startswith("--run=")]

    if "--all-problems" in args:
        args.remove("--all-problems")
        problem_paths = sorted(globmod.glob("problems/*.toml"))
        if not problem_paths:
            print("ERROR: No .toml files found in problems/", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(problem_paths)} problem files: {problem_paths}", file=sys.stderr)
    elif args:
        problem_paths = args
    else:
        problem_paths = ["problem.toml"]

    multi_mode = len(problem_paths) > 1

    # Load results once
    print(f"Loading results from {RESULTS_FILE}...", file=sys.stderr)
    all_rows = load_results()
    if not all_rows:
        print("ERROR: No results found in results.tsv", file=sys.stderr)
        sys.exit(1)

    runs = detect_runs(all_rows)
    print(f"Detected {len(runs)} run(s) in results.tsv", file=sys.stderr)

    # Collect data for each problem
    collected = []
    for pp in problem_paths:
        print(f"\nAnalysing {pp}...", file=sys.stderr)
        data = _collect_problem_data(pp, all_rows, runs)
        if data is not None:
            collected.append(data)

    if not collected:
        print("ERROR: No problems produced results.", file=sys.stderr)
        sys.exit(1)

    # Print terminal reports
    for data in collected:
        print(data["terminal_output"])

    # Generate HTML
    if multi_mode:
        # Combined tabbed report
        print("\nGenerating combined HTML report...", file=sys.stderr)
        problem_bodies = []
        problem_results = []
        for data in collected:
            body = _generate_problem_body_html(
                data["problem"], data["best_row"], data["rows"], data["evo_data"],
                data["feature_names"], data["importances"], data["imp_type"],
                data["interactions"], data["metric"], data["direction"],
                shap_plots=data.get("shap_plots"),
            )
            display_name = _pretty_problem_name(data["name"])
            problem_bodies.append((display_name, body))
            problem_results.append({
                "name": data["name"],
                "metric": data["metric"],
                "direction": data["direction"],
                "score": data["score"],
                "max_gen": data["max_gen"],
                "n_evals": data["n_evals"],
                "best_alg": data["best_alg"],
                "convergence_gen": data["convergence_gen"],
                "evo_data": data["evo_data"],
            })

        html = generate_combined_html_report(problem_bodies, problem_results)
        html_path = "report_combined.html"
        with open(html_path, "w") as f:
            f.write(html)
        print(f"Combined HTML report saved to: {html_path}", file=sys.stderr)
    else:
        # Single-problem report (original behaviour)
        data = collected[0]
        print("Generating HTML report...", file=sys.stderr)
        html = generate_html_report(
            data["problem"], data["best_row"], data["rows"], data["evo_data"],
            data["feature_names"], data["importances"], data["imp_type"],
            data["interactions"], data["metric"], data["direction"],
            shap_plots=data.get("shap_plots"),
        )
        problem_name = data["name"]
        html_path = f"report_{problem_name.replace(' ', '_').replace('-', '_')}.html"
        with open(html_path, "w") as f:
            f.write(html)
        print(f"HTML report saved to: {html_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
