"""
Dual-Lineage Runner: orchestrates two parallel evolutionary streams with
periodic Padgett/Wood-style block crossover.

This is the main entry point for the genetic AutoML system. It:
1. Maintains two independent lineages (A and B), each with its own genome
2. Each lineage independently mutates and evaluates via train.py
3. Every few generations, attempts a block crossover between lineages
4. Keeps improvements, discards regressions

Usage:
    python dual_runner.py                    # run with defaults
    python dual_runner.py --crossover-interval 3  # crossover every 3 generations
    python dual_runner.py --max-generations 50     # stop after 50 generations

The runner manages git commits so each experiment is reproducible.
Results are logged to results.tsv.
"""

import argparse
import copy
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from genome import Genome, GENE_DEFS
from genetic_algorithm import (
    Lineage, mutate, mutate_block, block_crossover, multi_block_crossover,
    select, should_crossover, BLOCK_NAMES, FUNCTIONAL_BLOCKS,
    CrossoverResult, SelectionDecision,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_SCRIPT = "train.py"
RESULTS_FILE = "results.tsv"
RUN_LOG = "run.log"
GENOME_DIR = "genomes"       # directory to save genome snapshots

# Markers in train.py that delimit the hyperparameters section
HP_START_MARKER = "# Hyperparameters (edit these directly, no CLI flags needed)"
HP_END_MARKER = "# Model size"
# Alternative: the actual variable assignments section
HP_SECTION_VARS = [
    "ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN",
    "TOTAL_BATCH_SIZE", "EMBEDDING_LR", "UNEMBEDDING_LR",
    "MATRIX_LR", "SCALAR_LR", "WEIGHT_DECAY", "ADAM_BETAS",
    "WARMUP_RATIO", "WARMDOWN_RATIO", "FINAL_LR_FRAC",
    "DEPTH", "DEVICE_BATCH_SIZE",
]

# Time budget for each training run (seconds)
# The actual budget is in prepare.py (300s), but we add buffer for
# startup/compilation/eval overhead
MAX_RUN_TIME = 600  # 10 minutes total wall clock (kill if exceeds)


# ---------------------------------------------------------------------------
# train.py manipulation: inject genome hyperparameters
# ---------------------------------------------------------------------------

def read_train_py() -> str:
    """Read the current train.py content."""
    return Path(TRAIN_SCRIPT).read_text()


def write_train_py(content: str):
    """Write new content to train.py."""
    Path(TRAIN_SCRIPT).write_text(content)


def extract_current_genome(content: str) -> Genome:
    """
    Parse the current hyperparameters from train.py into a Genome.

    Reads the actual variable assignments to reconstruct the genome.
    """
    genes = {}

    # Parse each known variable
    patterns = {
        "aspect_ratio": r"^ASPECT_RATIO\s*=\s*(\d+)",
        "head_dim": r"^HEAD_DIM\s*=\s*(\d+)",
        "window_pattern": r'^WINDOW_PATTERN\s*=\s*["\'](\w+)["\']',
        "total_batch_size_exp": r"^TOTAL_BATCH_SIZE\s*=\s*2\*\*(\d+)",
        "embedding_lr": r"^EMBEDDING_LR\s*=\s*([\d.]+)",
        "unembedding_lr": r"^UNEMBEDDING_LR\s*=\s*([\d.]+)",
        "matrix_lr": r"^MATRIX_LR\s*=\s*([\d.]+)",
        "scalar_lr": r"^SCALAR_LR\s*=\s*([\d.]+)",
        "weight_decay": r"^WEIGHT_DECAY\s*=\s*([\d.]+)",
        "warmup_ratio": r"^WARMUP_RATIO\s*=\s*([\d.]+)",
        "warmdown_ratio": r"^WARMDOWN_RATIO\s*=\s*([\d.]+)",
        "final_lr_frac": r"^FINAL_LR_FRAC\s*=\s*([\d.]+)",
        "depth": r"^DEPTH\s*=\s*(\d+)",
        "device_batch_size": r"^DEVICE_BATCH_SIZE\s*=\s*(\d+)",
    }

    # Adam betas need special handling
    adam_pattern = r"^ADAM_BETAS\s*=\s*\(([\d.]+),\s*([\d.]+)\)"

    for line in content.split("\n"):
        line = line.strip()
        for gene_name, pattern in patterns.items():
            m = re.match(pattern, line)
            if m:
                gdef = GENE_DEFS[gene_name]
                val = m.group(1)
                if gdef["type"] in ("int",):
                    genes[gene_name] = int(val)
                elif gdef["type"] in ("float", "logfloat"):
                    genes[gene_name] = float(val)
                elif gdef["type"] == "choice":
                    # Try to cast to the right type
                    choices = gdef["choices"]
                    if isinstance(choices[0], int):
                        genes[gene_name] = int(val)
                    elif isinstance(choices[0], bool):
                        genes[gene_name] = val.lower() == "true"
                    else:
                        genes[gene_name] = val

        # Adam betas
        m = re.match(adam_pattern, line)
        if m:
            genes["adam_beta1"] = float(m.group(1))
            genes["adam_beta2"] = float(m.group(2))

    # Fill in defaults for genes we couldn't parse
    for name, gdef in GENE_DEFS.items():
        if name not in genes:
            genes[name] = gdef["default"]

    return Genome(genes)


def inject_genome(content: str, genome: Genome) -> str:
    """
    Replace the hyperparameter section in train.py with values from a genome.

    Finds each variable assignment and replaces its value while preserving
    the surrounding code structure and comments.
    """
    g = genome.genes
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        stripped = line.strip()
        replaced = False

        # Try to replace each known variable
        replacements = {
            "ASPECT_RATIO": f"ASPECT_RATIO = {g['aspect_ratio']}",
            "HEAD_DIM": f"HEAD_DIM = {g['head_dim']}",
            "WINDOW_PATTERN": f'WINDOW_PATTERN = "{g["window_pattern"]}"',
            "TOTAL_BATCH_SIZE": f"TOTAL_BATCH_SIZE = 2**{g['total_batch_size_exp']}",
            "EMBEDDING_LR": f"EMBEDDING_LR = {g['embedding_lr']}",
            "UNEMBEDDING_LR": f"UNEMBEDDING_LR = {g['unembedding_lr']}",
            "MATRIX_LR": f"MATRIX_LR = {g['matrix_lr']}",
            "SCALAR_LR": f"SCALAR_LR = {g['scalar_lr']}",
            "WEIGHT_DECAY": f"WEIGHT_DECAY = {g['weight_decay']}",
            "ADAM_BETAS": f"ADAM_BETAS = ({g['adam_beta1']}, {g['adam_beta2']})",
            "WARMUP_RATIO": f"WARMUP_RATIO = {g['warmup_ratio']}",
            "WARMDOWN_RATIO": f"WARMDOWN_RATIO = {g['warmdown_ratio']}",
            "FINAL_LR_FRAC": f"FINAL_LR_FRAC = {g['final_lr_frac']}",
            "DEPTH": f"DEPTH = {g['depth']}",
            "DEVICE_BATCH_SIZE": f"DEVICE_BATCH_SIZE = {g['device_batch_size']}",
        }

        for var_name, replacement in replacements.items():
            if stripped.startswith(var_name) and "=" in stripped:
                # Preserve any inline comment
                comment = ""
                if "#" in stripped:
                    # Find the comment part (after the value assignment)
                    parts = stripped.split("#", 1)
                    # Only keep comment if it's after the value
                    if not parts[0].strip().startswith("#"):
                        comment = "  # " + parts[1].strip()

                # Preserve leading whitespace
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}{replacement}{comment}")
                replaced = True
                break

        if not replaced:
            new_lines.append(line)

    return "\n".join(new_lines)


# ---------------------------------------------------------------------------
# Run management: execute train.py and extract results
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a single training run."""
    success: bool
    val_bpb: float = 0.0
    peak_vram_mb: float = 0.0
    training_seconds: float = 0.0
    total_seconds: float = 0.0
    num_steps: int = 0
    num_params_M: float = 0.0
    error: str = ""


def run_training() -> RunResult:
    """
    Execute train.py and extract results.

    Returns a RunResult with the metrics or error information.
    """
    result = RunResult(success=False)

    try:
        proc = subprocess.run(
            ["uv", "run", TRAIN_SCRIPT],
            capture_output=True,
            text=True,
            timeout=MAX_RUN_TIME,
        )

        # Write log
        log_content = proc.stdout + "\n" + proc.stderr
        Path(RUN_LOG).write_text(log_content)

        if proc.returncode != 0:
            # Training crashed
            result.error = _extract_error(log_content)
            return result

        # Parse metrics from output
        result = _parse_metrics(log_content)
        return result

    except subprocess.TimeoutExpired:
        result.error = f"Training exceeded {MAX_RUN_TIME}s timeout"
        return result
    except Exception as e:
        result.error = str(e)
        return result


def _parse_metrics(log: str) -> RunResult:
    """Parse the metrics block from train.py output."""
    result = RunResult(success=True)

    patterns = {
        "val_bpb": r"^val_bpb:\s+([\d.]+)",
        "peak_vram_mb": r"^peak_vram_mb:\s+([\d.]+)",
        "training_seconds": r"^training_seconds:\s+([\d.]+)",
        "total_seconds": r"^total_seconds:\s+([\d.]+)",
        "num_steps": r"^num_steps:\s+(\d+)",
        "num_params_M": r"^num_params_M:\s+([\d.]+)",
    }

    for line in log.split("\n"):
        line = line.strip()
        for key, pattern in patterns.items():
            m = re.match(pattern, line)
            if m:
                val = float(m.group(1))
                if key == "num_steps":
                    result.num_steps = int(val)
                else:
                    setattr(result, key, val)

    # If we didn't find val_bpb, it's a failure
    if result.val_bpb == 0.0:
        result.success = False
        result.error = "No val_bpb found in output"

    return result


def _extract_error(log: str) -> str:
    """Extract the last error/traceback from a log."""
    lines = log.strip().split("\n")
    # Look for traceback
    for i in range(len(lines) - 1, -1, -1):
        if "Error" in lines[i] or "error" in lines[i]:
            return lines[i][:200]
    # Fall back to last few lines
    return "\n".join(lines[-5:])[:500]


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------

def git_commit(message: str) -> str:
    """Commit current changes and return the short hash."""
    subprocess.run(["git", "add", TRAIN_SCRIPT], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True,
                   capture_output=True)
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True)
    return result.stdout.strip()


def git_reset_to(commit_hash: str):
    """Reset to a specific commit (discarding current changes)."""
    subprocess.run(["git", "reset", "--hard", commit_hash], check=True,
                   capture_output=True)


def git_current_hash() -> str:
    """Get current short commit hash."""
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True)
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

def init_results_tsv():
    """Create results.tsv with header if it doesn't exist."""
    if not Path(RESULTS_FILE).exists():
        Path(RESULTS_FILE).write_text(
            "commit\tval_bpb\tmemory_gb\tstatus\tlineage\tgeneration\tdescription\n"
        )


def log_result(commit: str, val_bpb: float, memory_gb: float,
               status: str, lineage: str, generation: int, description: str):
    """Append a result to results.tsv."""
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t"
                f"{lineage}\t{generation}\t{description}\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiment(lineage: Lineage, genome: Genome, description: str,
                   dry_run: bool = False) -> RunResult:
    """
    Run a single experiment: inject genome -> train -> extract results.

    Returns the RunResult and updates train.py.
    """
    # Inject the genome into train.py
    content = read_train_py()
    new_content = inject_genome(content, genome)
    write_train_py(new_content)

    if dry_run:
        print(f"  [DRY RUN] Would train with: {genome.describe()}")
        return RunResult(success=True, val_bpb=random.uniform(0.9, 1.1))

    # Commit the change
    commit = git_commit(f"[{lineage.name}] gen{lineage.generation}: {description}")

    # Run training
    print(f"  Training... (this takes ~5 minutes)")
    t0 = time.time()
    result = run_training()
    elapsed = time.time() - t0

    # Log results
    memory_gb = result.peak_vram_mb / 1024 if result.success else 0.0
    if result.success:
        status = "pending"  # will be updated to keep/discard
        print(f"  val_bpb={result.val_bpb:.6f} | vram={memory_gb:.1f}GB | "
              f"steps={result.num_steps} | {elapsed:.0f}s")
    else:
        status = "crash"
        print(f"  CRASH: {result.error[:100]}")
        log_result(commit, 0.0, 0.0, "crash", lineage.name,
                  lineage.generation, description)

    return result


def run_dual_lineage(
    max_generations: int = 100,
    crossover_interval: int = 5,
    mutation_rate: float = 0.3,
    mutation_strength: float = 0.3,
    seed: int = 42,
    dry_run: bool = False,
):
    """
    Main loop: evolve two lineages with periodic Padgett/Wood crossover.

    The algorithm:
    1. Initialize two lineages with the baseline genome
    2. Run baseline evaluation for both
    3. Loop:
       a. If it's crossover time -> attempt block crossover between lineages
       b. Otherwise -> mutate the active lineage's genome
       c. Evaluate the candidate
       d. Keep if improved, discard if not
       e. Alternate between lineages
    """
    rng = random.Random(seed)
    init_results_tsv()

    # Save starting point
    start_hash = git_current_hash()
    print(f"Starting from commit: {start_hash}")

    # Parse the baseline genome from current train.py
    baseline_genome = extract_current_genome(read_train_py())
    print(f"Baseline genome: {baseline_genome.describe()}")

    # Initialize two lineages
    lineage_a = Lineage(name="A", genome=baseline_genome.copy())
    lineage_b = Lineage(name="B", genome=baseline_genome.copy())

    # --- Baseline run ---
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)

    baseline_result = run_experiment(lineage_a, baseline_genome, "baseline", dry_run)
    if baseline_result.success:
        lineage_a.fitness = baseline_result.val_bpb
        lineage_b.fitness = baseline_result.val_bpb
        lineage_a.record(baseline_result.val_bpb, "baseline")
        lineage_b.record(baseline_result.val_bpb, "baseline (shared)")
        log_result(git_current_hash(), baseline_result.val_bpb,
                  baseline_result.peak_vram_mb / 1024, "keep",
                  "baseline", 0, "baseline")
        print(f"\nBaseline val_bpb: {baseline_result.val_bpb:.6f}")
    else:
        print("FATAL: Baseline run failed. Fix train.py before running the GA.")
        sys.exit(1)

    # Track the last-good commit for each lineage
    a_good_hash = git_current_hash()
    b_good_hash = a_good_hash

    # --- Main evolution loop ---
    lineages = [lineage_a, lineage_b]
    active_idx = 0  # alternate between lineages

    for gen in range(1, max_generations + 1):
        active = lineages[active_idx]
        other = lineages[1 - active_idx]
        good_hash = a_good_hash if active_idx == 0 else b_good_hash

        print(f"\n{'=' * 60}")
        print(f"GENERATION {gen} | Lineage {active.name} "
              f"(fitness={active.fitness:.6f}) | "
              f"Other={other.name} ({other.fitness:.6f})")
        print(f"{'=' * 60}")

        # Decide: crossover or mutation?
        do_crossover = should_crossover(gen, crossover_interval, rng)

        if do_crossover and other.fitness < float("inf"):
            # --- CROSSOVER ---
            # Randomly choose single or multi-block
            if rng.random() < 0.7:
                xover = block_crossover(active.genome, other.genome, rng=rng)
            else:
                xover = multi_block_crossover(active.genome, other.genome,
                                              num_blocks=2, rng=rng)
            xover.recipient_lineage = active.name
            xover.donor_lineage = other.name

            candidate = xover.child
            description = f"crossover({xover.block_name}) from {other.name}"
            print(f"  Attempting crossover: {xover.block_name} from {other.name}")
            print(f"  Transplanted genes: {list(xover.block_genes.keys())}")

        else:
            # --- MUTATION ---
            # Mix of gene-level and block-level mutation
            if rng.random() < 0.6:
                candidate = mutate(active.genome, mutation_rate, mutation_strength, rng)
                description = f"mutate({len(active.genome.diff(candidate))} genes)"
            else:
                block = rng.choice(BLOCK_NAMES)
                candidate = mutate_block(active.genome, block, mutation_strength, rng)
                description = f"mutate_block({block})"

            diffs = active.genome.diff(candidate)
            print(f"  Mutating: {description}")
            if diffs:
                for name, (old, new) in list(diffs.items())[:5]:
                    print(f"    {name}: {old} -> {new}")
                if len(diffs) > 5:
                    print(f"    ... and {len(diffs) - 5} more")

        # Reset to the lineage's last good state before injecting new genome
        git_reset_to(good_hash)

        # Run the experiment
        result = run_experiment(active, candidate, description, dry_run)

        if not result.success:
            active.crash(description)
            git_reset_to(good_hash)
            print(f"  -> CRASH, reverting to {good_hash}")
            active_idx = 1 - active_idx
            continue

        # --- SELECTION ---
        if do_crossover and other.fitness < float("inf"):
            # Padgett/Wood three-way decision
            decision = select(result.val_bpb, active.fitness, other.fitness,
                            candidate)
            print(f"  -> {decision.description}")

            if decision.action == "keep_as_recipient":
                commit = git_current_hash()
                active.advance(candidate, result.val_bpb, description)
                if active_idx == 0:
                    a_good_hash = commit
                else:
                    b_good_hash = commit
                log_result(commit, result.val_bpb,
                          result.peak_vram_mb / 1024, "keep",
                          active.name, gen, description)
                print(f"  -> KEEP as {active.name}")

            elif decision.action == "adopt_as_donor":
                # The child helps the other lineage
                commit = git_current_hash()
                other.advance(candidate, result.val_bpb,
                            f"adopted from {active.name}: {description}")
                if active_idx == 0:
                    b_good_hash = commit
                else:
                    a_good_hash = commit
                log_result(commit, result.val_bpb,
                          result.peak_vram_mb / 1024, "keep",
                          other.name, gen,
                          f"adopted from {active.name}: {description}")
                git_reset_to(good_hash)  # revert active lineage
                print(f"  -> ADOPTED by {other.name}")

            else:
                git_reset_to(good_hash)
                active.reject(result.val_bpb, description)
                log_result(git_current_hash(), result.val_bpb,
                          result.peak_vram_mb / 1024, "discard",
                          active.name, gen, description)
                print(f"  -> DISCARD")

        else:
            # Simple keep/discard for mutations
            if result.val_bpb < active.fitness:
                commit = git_current_hash()
                improvement = active.fitness - result.val_bpb
                active.advance(candidate, result.val_bpb, description)
                if active_idx == 0:
                    a_good_hash = commit
                else:
                    b_good_hash = commit
                log_result(commit, result.val_bpb,
                          result.peak_vram_mb / 1024, "keep",
                          active.name, gen, description)
                print(f"  -> KEEP (improved by {improvement:.6f})")
            else:
                git_reset_to(good_hash)
                active.reject(result.val_bpb, description)
                log_result(git_current_hash(), result.val_bpb,
                          result.peak_vram_mb / 1024, "discard",
                          active.name, gen, description)
                print(f"  -> DISCARD")

        # Alternate lineages
        active_idx = 1 - active_idx

        # Status report
        print(f"\n  Status: A={lineage_a.fitness:.6f} (gen {lineage_a.generation}) | "
              f"B={lineage_b.fitness:.6f} (gen {lineage_b.generation})")

    # --- Final report ---
    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"\nLineage A: val_bpb={lineage_a.fitness:.6f} after {lineage_a.generation} generations")
    print(f"Lineage B: val_bpb={lineage_b.fitness:.6f} after {lineage_b.generation} generations")
    best = lineage_a if lineage_a.fitness <= lineage_b.fitness else lineage_b
    print(f"\nBest: Lineage {best.name} with val_bpb={best.fitness:.6f}")
    print(f"Config: {best.genome.describe()}")

    # Save best genome
    os.makedirs(GENOME_DIR, exist_ok=True)
    best.genome.save(os.path.join(GENOME_DIR, "best_genome.json"))
    lineage_a.genome.save(os.path.join(GENOME_DIR, "lineage_a.json"))
    lineage_b.genome.save(os.path.join(GENOME_DIR, "lineage_b.json"))
    print(f"\nGenomes saved to {GENOME_DIR}/")

    return best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dual-lineage genetic AutoML with Padgett/Wood crossover"
    )
    parser.add_argument("--max-generations", type=int, default=100,
                       help="Maximum generations to run (default: 100)")
    parser.add_argument("--crossover-interval", type=int, default=5,
                       help="Attempt crossover every N generations (default: 5)")
    parser.add_argument("--mutation-rate", type=float, default=0.3,
                       help="Probability of mutating each gene (default: 0.3)")
    parser.add_argument("--mutation-strength", type=float, default=0.3,
                       help="How far mutations can go (0-1, default: 0.3)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't actually train, use random fitness values")
    args = parser.parse_args()

    run_dual_lineage(
        max_generations=args.max_generations,
        crossover_interval=args.crossover_interval,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        seed=args.seed,
        dry_run=args.dry_run,
    )
