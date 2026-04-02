# autoresearch

An autonomous AutoML framework that evolves ML pipelines to optimize any metric on tabular datasets. Inspired by [Karpathy's autoresearch](https://x.com/karpathy/status/2029701092347630069) — the same "time-limited experiments with autonomous iteration" philosophy, generalized from LLM tuning to the full ML pipeline space.

## How it works

An evolutionary search engine explores combinations of preprocessing, feature selection, and ML algorithms with their hyperparameters using an **island model** — multiple sub-populations evolve in parallel and periodically exchange their best individuals to maximize diversity. An AI agent iterates on the search space definition (`search_space.py`) to find the best pipeline for your problem.

**Key files:**

```
problem.toml      — problem definition: dataset, target, metric, time budget
prepare.py        — data loading, splitting, evaluation harness (read-only)
pipeline.py       — pipeline config, construction, execution (read-only)
search_space.py   — operator registries, hyperparameter ranges (AGENT EDITS THIS)
evolve.py         — evolutionary search engine (read-only)
program.md        — agent instructions
```

**The search space includes:** StandardScaler, MinMaxScaler, RobustScaler, PCA, PolynomialFeatures, SelectKBest, RandomForest, GradientBoosting, ExtraTrees, XGBoost, LightGBM, Ridge, Lasso, ElasticNet, SVR, KNeighbors, DecisionTree, AdaBoost, and MLP (via PyTorch).

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Configure your problem in problem.toml, then run
uv run evolve.py
```

### Data sources

**Local file (CSV/Parquet):**
```toml
[data]
source = "file"
train = "data/train.parquet"
target = "SalePrice"
```

**Built-in sklearn dataset:**
```toml
[data]
source = "sklearn"
dataset = "california_housing"
target = "MedHouseVal"
```

**Snowflake query:**
```toml
[data]
source = "snowflake"
query = "SELECT * FROM db.schema.table"
target = "target_col"
connection = "default"
```

## Running the agent

Point your AI agent at this repo and prompt:

```
Read program.md and let's kick off a new experiment!
```

The agent will modify `search_space.py`, run `evolve.py`, check results, keep or discard changes, and repeat autonomously.

## Output format

```
---
best_score:           0.523456
metric:               rmse
generations:          47
pipelines_evaluated:  940
total_seconds:        305.2
best_pipeline:        StandardScaler -> SelectKBest(k=15) -> XGBoost(n_estimators=200, max_depth=8)
```

## Design choices

- **Single file to modify.** The agent only edits `search_space.py` — operator registries, hyperparameter ranges, and evolution parameters.
- **Fixed time budget.** Evolution runs for a configurable duration (default 5 minutes). All runs are directly comparable.
- **Evolutionary search.** Tournament selection with block swap crossover naturally handles the mixed structure+hyperparameter search space.
- **Island model parallelism.** Multiple sub-populations evolve concurrently via `ThreadPoolExecutor`. Configure `n_workers` in `problem.toml` (defaults to `cpu_count // 2`, max 8). Migration broadcasts the global best to all islands every 15 seconds.
- **GPU optional.** CPU for sklearn/xgboost/lightgbm. GPU only used if the MLP algorithm is selected and CUDA is available.

## Supported metrics

| Metric | Direction | Task |
|--------|-----------|------|
| rmse | minimize | regression |
| mse | minimize | regression |
| mae | minimize | regression |
| r2 | maximize | regression |
| adj_r2 | maximize | regression |
| auc | maximize | classification |
| logloss | minimize | classification |
| accuracy | maximize | classification |
| f1 | maximize | classification |

## License

MIT
