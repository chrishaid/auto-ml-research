# autoresearch — AutoML Evolution

This is an AutoML framework that searches over ML pipelines (preprocessing, feature selection, algorithms, hyperparameters) to minimize a configurable loss function on arbitrary tabular datasets.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar17`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these for full context:
   - `README.md` — repository context.
   - `problem.toml` — problem definition (dataset, target, metric, time budget).
   - `prepare.py` — data loading, splitting, evaluation harness. **Do not modify.**
   - `pipeline.py` — pipeline config, construction, execution. **Do not modify.**
   - `search_space.py` — **the file you modify.** Operator registries, hyperparameter ranges, evolution parameters.
   - `evolve.py` — evolutionary search engine. **Do not modify.**
4. **Verify problem.toml**: Ensure the dataset is accessible (file exists, sklearn dataset available, or Snowflake connection works).
5. **Initialize results.tsv**: Will be auto-created on first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs `uv run evolve.py`. The evolution runs for a **fixed time budget** (default 5 minutes, configurable in `problem.toml`). The engine uses an **island model** with parallel sub-populations (configurable via `n_workers` in `problem.toml`). Islands evolve independently and periodically exchange their best individuals via migration to maintain diversity.

**What you CAN do:**
- Modify `search_space.py` — this is the only file you edit. You can:
  - Add/remove/modify algorithms in `ALGORITHMS`
  - Add/remove/modify preprocessors in `PREPROCESSORS`
  - Add/remove/modify feature selectors in `FEATURE_SELECTORS`
  - Tune hyperparameter ranges (narrower/wider/different distributions)
  - Adjust evolution parameters (`POPULATION_SIZE`, `MUTATION_RATE`, etc.)
  - Add custom sklearn-compatible estimators
  - Improve mutation/crossover operators

**What you CANNOT do:**
- Modify `prepare.py`, `pipeline.py`, or `evolve.py`. They are read-only.
- Install new packages beyond what's in `pyproject.toml`.
- Modify the evaluation harness.

**The goal: get the best score on the configured metric.** The direction (minimize/maximize) is defined in `problem.toml`. The evolution engine handles the search — your job is to give it the best search space.

**Simplicity criterion**: All else being equal, simpler is better. A marginal improvement that adds ugly complexity is not worth it. Removing operators that hurt performance and getting equal or better results is a simplification win.

## Output format

When evolve.py finishes it prints:

```
---
best_score:           0.523456
metric:               rmse
generations:          47
pipelines_evaluated:  940
total_seconds:        305.2
best_pipeline:        StandardScaler -> SelectKBest(k=15) -> XGBoost(n_estimators=200, max_depth=8)
```

Extract the key metric:
```
grep "^best_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

The TSV has a header row and 5 columns:

```
commit	best_score	metric	status	description
```

1. git commit hash (short, 7 chars)
2. best_score achieved (e.g. 0.523456)
3. metric name
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried (changes to search_space.py)

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar17`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit
2. Modify `search_space.py` with a hypothesis (new algorithm, different hyperparameter ranges, evolution params, etc.)
3. git commit
4. Run: `uv run evolve.py > run.log 2>&1`
5. Read results: `grep "^best_score:\|^best_pipeline:" run.log`
6. If grep output is empty, the run crashed. Run `tail -n 50 run.log` for the traceback.
7. Record results in the TSV (do NOT commit results.tsv)
8. If best_score improved, keep the commit
9. If best_score is equal or worse, git reset back

**Timeout**: Each run should take ~5 minutes (configurable). If it exceeds 2x the budget, kill it.

**Crashes**: Fix trivial bugs and re-run. Skip fundamentally broken ideas.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask the human. You are autonomous. If you run out of ideas, think harder — try different algorithm combinations, narrow hyperparameter ranges based on results, add domain-specific preprocessors, try ensembles. The loop runs until the human interrupts.
