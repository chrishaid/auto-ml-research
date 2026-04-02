"""
Evolutionary search engine for AutoML pipeline optimization.

Main entry point: `uv run evolve.py`

Reads problem.toml, initializes a population of random pipelines,
and evolves them using an island model with parallel sub-populations.
Each island runs tournament selection, block swap crossover, and mutation.
The best individual migrates between islands on a ring topology.

Output format (printed at end):
---
best_score:           <float>
metric:               <metric_name>
generations:          <int>
pipelines_evaluated:  <int>
total_seconds:        <float>
best_pipeline:        <human-readable pipeline description>
"""

import os
import sys
import csv
import time
import copy
import random
import threading
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

from prepare import load_problem, load_data, split_data, auto_preprocess, evaluate, score_to_fitness
from pipeline import PipelineConfig, build_sklearn_pipeline, execute_pipeline
from search_space import (
    get_registry, random_pipeline, mutate, crossover,
    POPULATION_SIZE, OFFSPRING_PER_GEN, TOURNAMENT_SIZE,
    MUTATION_RATE, CROSSOVER_RATE, MIGRATION_INTERVAL,
    MIN_EVALUATIONS,
)

# Suppress sklearn warnings during evolution
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

RESULTS_FILE = "results.tsv"
_results_lock = threading.Lock()


def init_results():
    """Initialize results.tsv with header if it doesn't exist."""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["generation", "pipeline_id", "score", "fitness", "elapsed_s", "status", "description"])


def log_result(generation, pipeline_id, score, fitness, elapsed, status, description):
    """Append one result row to results.tsv (thread-safe)."""
    with _results_lock:
        with open(RESULTS_FILE, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            score_str = f"{score:.6f}" if score is not None else "N/A"
            fitness_str = f"{fitness:.6f}" if fitness is not None else "N/A"
            writer.writerow([generation, pipeline_id, score_str, fitness_str, f"{elapsed:.1f}", status, description])


# ---------------------------------------------------------------------------
# Tournament selection
# ---------------------------------------------------------------------------

def tournament_select(population, fitnesses, k=TOURNAMENT_SIZE):
    """Select one individual via tournament selection (higher fitness = better)."""
    indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


# ---------------------------------------------------------------------------
# Shared state for island model
# ---------------------------------------------------------------------------

@dataclass
class SharedState:
    """Thread-safe shared state for inter-island communication."""
    inboxes: list = field(default_factory=list)        # one inbox (list) per island
    inbox_locks: list = field(default_factory=list)     # one lock per inbox
    best_fitness: float = float("-inf")                 # global best fitness
    best_config: PipelineConfig = None                  # global best config
    best_lock: threading.Lock = field(default_factory=threading.Lock)
    total_evaluated: list = field(default_factory=list)  # per-island counters
    total_generations: list = field(default_factory=list)  # per-island gen counters

    @classmethod
    def create(cls, n_islands):
        state = cls()
        state.inboxes = [[] for _ in range(n_islands)]
        state.inbox_locks = [threading.Lock() for _ in range(n_islands)]
        state.total_evaluated = [0] * n_islands
        state.total_generations = [0] * n_islands
        return state

    def update_global_best(self, fitness, config):
        """Update global best if this fitness is better. Returns True if updated."""
        with self.best_lock:
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_config = copy.deepcopy(config)
                return True
            return False

    def push_migrant(self, island_id, config):
        """Push a migrant into an island's inbox."""
        with self.inbox_locks[island_id]:
            self.inboxes[island_id].append(copy.deepcopy(config))

    def pop_migrant(self, island_id):
        """Pop a migrant from an island's inbox (or None)."""
        with self.inbox_locks[island_id]:
            if self.inboxes[island_id]:
                return self.inboxes[island_id].pop(0)
            return None


# ---------------------------------------------------------------------------
# Single island evolution
# ---------------------------------------------------------------------------

def evolve_island(island_id, shared_state, n_features, registry, task_type,
                  metric_name, pipeline_timeout, time_budget, min_evaluations,
                  t_start, X_train_np, y_train_np, X_val_np, y_val_np):
    """Run evolution on a single island. Called in a thread."""

    def should_stop():
        """Hybrid stopping: time must be up AND min evals reached across all islands."""
        elapsed = time.time() - t_start
        total_evals = sum(shared_state.total_evaluated)
        if elapsed >= time_budget and total_evals >= min_evaluations:
            return True
        # Hard cap: 2x time budget regardless of eval count
        if elapsed >= time_budget * 2:
            return True
        return False

    # Unique seed per island for diversity
    island_seed = 42 + island_id * 1000
    rng = random.Random(island_seed)
    np_rng = np.random.RandomState(island_seed)

    # Override random/np.random for this thread via a local helper
    # We use the module-level random for search_space operators, so we seed it per-island
    # This is imperfect with threads but provides good diversity in practice
    random.seed(island_seed)
    np.random.seed(island_seed)

    def eval_pipeline(config, gen, pid):
        desc = config.describe()
        try:
            score, elapsed, error = execute_pipeline(
                config, registry, task_type,
                X_train_np, y_train_np, X_val_np, y_val_np,
                metric_name, timeout=pipeline_timeout,
            )
            if error:
                log_result(gen, pid, None, None, elapsed, "error", f"[I{island_id}] {desc} | {error[:80]}")
                return None, desc
            fitness = score_to_fitness(score, metric_name)
            log_result(gen, pid, score, fitness, elapsed, "ok", f"[I{island_id}] {desc}")
            return fitness, desc
        except Exception:
            log_result(gen, pid, None, None, 0.0, "crash", f"[I{island_id}] {desc} | {traceback.format_exc()[:80]}")
            return None, desc

    # --- Initialize population ---
    population = []
    fitnesses = []
    local_pid = 0

    for i in range(POPULATION_SIZE):
        if should_stop():
            break
        config = random_pipeline(n_features)
        fitness, desc = eval_pipeline(config, 0, f"I{island_id}-{local_pid}")
        if fitness is not None:
            population.append(config)
            fitnesses.append(fitness)
            print(f"  Island {island_id} [{local_pid:04d}] fitness={fitness:+.6f}  {desc}")
        else:
            population.append(config)
            fitnesses.append(float("-inf"))
            print(f"  Island {island_id} [{local_pid:04d}] FAILED  {desc}")
        local_pid += 1
        shared_state.total_evaluated[island_id] += 1

    if not population:
        print(f"  Island {island_id}: Could not initialize any pipelines.")
        return

    # Track local best
    best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    local_best_fitness = fitnesses[best_idx]
    local_best_config = population[best_idx]

    # Update global best
    shared_state.update_global_best(local_best_fitness, local_best_config)

    print(f"  Island {island_id} initialized: best={local_best_fitness:+.6f}  {local_best_config.describe()}")

    # --- Evolution loop ---
    generation = 1

    while True:
        if should_stop():
            break

        gen_improved = False

        for _ in range(OFFSPRING_PER_GEN):
            if should_stop():
                break

            # Generate offspring
            if random.random() < CROSSOVER_RATE and len(population) >= 2:
                parent_a = tournament_select(population, fitnesses)
                parent_b = tournament_select(population, fitnesses)
                child = crossover(parent_a, parent_b)
            else:
                parent = tournament_select(population, fitnesses)
                child = mutate(parent, n_features)

            # Apply mutation to crossover offspring too
            if random.random() < MUTATION_RATE:
                child = mutate(child, n_features)

            # Evaluate
            fitness, desc = eval_pipeline(child, generation, f"I{island_id}-{local_pid}")
            local_pid += 1
            shared_state.total_evaluated[island_id] += 1

            if fitness is None:
                continue

            # Steady-state replacement: replace worst if child is better
            worst_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
            if fitness > fitnesses[worst_idx]:
                population[worst_idx] = child
                fitnesses[worst_idx] = fitness

            # Update local + global best
            if fitness > local_best_fitness:
                local_best_fitness = fitness
                local_best_config = child
                gen_improved = True
                is_global = shared_state.update_global_best(fitness, child)
                marker = " ** GLOBAL BEST **" if is_global else ""
                print(f"  Island {island_id} Gen {generation:04d} NEW BEST fitness={fitness:+.6f}  {desc}{marker}")

        # Check inbox for migrants
        migrant = shared_state.pop_migrant(island_id)
        if migrant is not None:
            # Evaluate the migrant in our context
            migrant_fitness, migrant_desc = eval_pipeline(migrant, generation, f"I{island_id}-mig")
            shared_state.total_evaluated[island_id] += 1
            if migrant_fitness is not None:
                worst_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
                if migrant_fitness > fitnesses[worst_idx]:
                    population[worst_idx] = migrant
                    fitnesses[worst_idx] = migrant_fitness
                    if migrant_fitness > local_best_fitness:
                        local_best_fitness = migrant_fitness
                        local_best_config = migrant

        remaining = max(0, time_budget - (time.time() - t_start))
        avg_fitness = np.mean([f for f in fitnesses if f > float("-inf")])
        print(f"Island {island_id} Gen {generation:04d} | best={local_best_fitness:+.6f} | avg={avg_fitness:+.6f} | evaluated={shared_state.total_evaluated[island_id]} | remaining={remaining:.0f}s")

        generation += 1
        shared_state.total_generations[island_id] = generation - 1


# ---------------------------------------------------------------------------
# Migration coordinator
# ---------------------------------------------------------------------------

def migrate(shared_state, n_islands, island_populations):
    """
    Ring topology migration: copy best from island i to island (i+1) % n_islands inbox.
    Called by the main thread on a timer.
    """
    # We don't have direct access to island populations from the main thread,
    # so migration is handled via the shared_state best tracking per island.
    # Instead, we use a simpler approach: the main thread reads each island's
    # inbox and pushes the global best to all islands periodically.
    # This is a broadcast migration strategy.
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    random.seed(42)
    np.random.seed(42)

    # Load problem
    problem_path = sys.argv[1] if len(sys.argv) > 1 else "problem.toml"
    problem = load_problem(problem_path)
    n_workers = problem["n_workers"]
    print(f"Problem: {problem['name']} ({problem['task']})")
    print(f"Metric: {problem['metric']} ({problem['direction']})")
    min_evaluations = problem["min_evaluations"]
    print(f"Time budget: {problem['time_budget']}s")
    print(f"Min evaluations: {min_evaluations}")
    print(f"Pipeline timeout: {problem['pipeline_timeout']}s")
    print(f"Islands: {n_workers}")
    print()

    # Load and preprocess data
    print("Loading data...")
    X, y = load_data(problem)
    X = auto_preprocess(X)
    X_train, X_val, y_train, y_val = split_data(X, y)
    n_features = X_train.shape[1]
    print(f"Data: {X_train.shape[0]} train, {X_val.shape[0]} val, {n_features} features")
    print()

    # Convert to numpy for sklearn
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    y_val_np = y_val.values if hasattr(y_val, 'values') else y_val

    registry = get_registry()
    task_type = problem["task"]
    metric_name = problem["metric"]
    pipeline_timeout = problem["pipeline_timeout"]
    time_budget = problem["time_budget"]

    init_results()

    shared_state = SharedState.create(n_workers)

    # --- Launch islands ---
    print(f"Launching {n_workers} island(s)...")
    print()

    if n_workers == 1:
        # Single island — run directly, no thread pool overhead
        evolve_island(
            0, shared_state, n_features, registry, task_type,
            metric_name, pipeline_timeout, time_budget, min_evaluations,
            t_start, X_train_np, y_train_np, X_val_np, y_val_np,
        )
    else:
        # Multi-island with migration
        # We use a ring topology migration coordinated by the main thread
        # Each island stores its best in shared_state; the main thread
        # copies best from island i → inbox of island (i+1) % n_workers

        # Track per-island best for migration (updated via polling)
        island_best_fitness = [float("-inf")] * n_workers
        island_best_config = [None] * n_workers

        futures = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for i in range(n_workers):
                future = executor.submit(
                    evolve_island,
                    i, shared_state, n_features, registry, task_type,
                    metric_name, pipeline_timeout, time_budget, min_evaluations,
                    t_start, X_train_np, y_train_np, X_val_np, y_val_np,
                )
                futures[future] = i

            # Migration loop in main thread
            last_migration = time.time()
            while True:
                time.sleep(1)  # Check every second

                # Check if all futures done
                all_done = all(f.done() for f in futures)
                if all_done:
                    break

                # Hard cap: 2x time budget (matches island should_stop)
                if time.time() - t_start >= time_budget * 2 + 5:
                    break

                # Migration timer
                if time.time() - last_migration >= MIGRATION_INTERVAL:
                    last_migration = time.time()

                    # Ring migration: push global best to each island
                    # More sophisticated: track per-island best and do ring
                    # For now, broadcast global best to all islands
                    with shared_state.best_lock:
                        global_best = shared_state.best_config
                        global_best_fitness = shared_state.best_fitness

                    if global_best is not None:
                        for i in range(n_workers):
                            shared_state.push_migrant(i, global_best)
                        print(f"\n>> Migration: broadcasting global best (fitness={global_best_fitness:+.6f}) to all {n_workers} islands\n")

            # Collect any exceptions
            for future, island_id in futures.items():
                try:
                    future.result()
                except Exception as e:
                    print(f"Island {island_id} crashed: {e}")

    # --- Final output ---
    t_end = time.time()
    total_evaluated = sum(shared_state.total_evaluated)
    total_generations = max(shared_state.total_generations) if shared_state.total_generations else 0

    best_fitness = shared_state.best_fitness
    best_config = shared_state.best_config

    if best_config is None:
        print("ERROR: No valid pipelines found across all islands.")
        sys.exit(1)

    # Recover actual score from fitness
    direction = problem["direction"]
    if direction == "minimize":
        best_score = -best_fitness
    else:
        best_score = best_fitness

    print()
    print("---")
    print(f"best_score:           {best_score:.6f}")
    print(f"metric:               {metric_name}")
    print(f"generations:          {total_generations}")
    print(f"pipelines_evaluated:  {total_evaluated}")
    print(f"total_seconds:        {t_end - t_start:.1f}")
    print(f"best_pipeline:        {best_config.describe()}")

    # Full config dump
    print()
    print("best_config:")
    for name, params in best_config.preparation:
        print(f"  preparation:        {name}")
        for k, v in params.items():
            print(f"    {k}: {v!r}")
    for name, params in best_config.preprocessing:
        print(f"  preprocessing:      {name}")
        for k, v in params.items():
            print(f"    {k}: {v!r}")
    fs_name, fs_params = best_config.feature_selection
    print(f"  feature_selection:  {fs_name}")
    for k, v in fs_params.items():
        print(f"    {k}: {v!r}")
    alg_name, alg_params = best_config.algorithm
    print(f"  algorithm:          {alg_name}")
    for k, v in alg_params.items():
        print(f"    {k}: {v!r}")


if __name__ == "__main__":
    main()
