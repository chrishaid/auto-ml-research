# automl-evolve

Genetic AutoML with Padgett/Wood-style block crossover. Forked from [autoresearch](https://github.com/karpathy/autoresearch).

## Core Idea

Two model lineages evolve independently, each running the autoresearch loop (mutate → train 5 min → evaluate → keep/discard). Every few generations, a **functional block** (a coherent group of hyperparameters/architecture choices) is swapped from one lineage into the other — the Padgett & Wood "code crossover." If the hybrid beats either parent, it survives. Otherwise, revert.

This mirrors how organizations innovate: not by tuning individual parameters, but by transplanting entire *routines* between entities and seeing what sticks.

## Architecture

```
genome.py              — Gene definitions + Genome class (the DNA of an ML config)
genetic_algorithm.py   — Mutation, crossover, selection logic
dual_runner.py         — Orchestrates two parallel lineages (main entry point)
train.py               — The training script (modified by the GA, same as autoresearch)
prepare.py             — Fixed data prep, tokenizer, evaluation (DO NOT MODIFY)
```

## Functional Blocks

Crossover doesn't swap individual genes — it swaps *functional blocks*: coherent subsystems that work as a unit.

| Block | Genes | Description |
|-------|-------|-------------|
| `attention` | head_dim, window_pattern, use_value_embeds | Attention mechanism |
| `model_scale` | depth, aspect_ratio, device_batch_size, total_batch_size_exp | Model size + batch |
| `learning_rates` | embedding_lr, unembedding_lr, matrix_lr, scalar_lr | LR per param group |
| `optimizer_dynamics` | weight_decay, adam_beta1, adam_beta2, muon_ns_steps | Optimizer tuning |
| `lr_schedule` | warmup_ratio, warmdown_ratio, final_lr_frac | Schedule shape |
| `mlp` | mlp_expansion, activation | MLP architecture |
| `residual` | softcap, x0_lambda_init | Residual/output scaling |

## The Crossover Decision (Padgett/Wood Rule)

```
child = recipient_genome + donor_block    # transplant one functional block

child_bpb = train_and_evaluate(child)     # 5-minute training run

if child_bpb < recipient_bpb:
    recipient adopts child                # the transplant helped the recipient
elif child_bpb < donor_bpb:
    donor adopts child                    # the transplant helped the donor instead
else:
    discard child                         # didn't help anyone, revert
```

## Setup

1. **Install dependencies**: `uv sync`
2. **Prepare data**: `uv run prepare.py` (one-time, downloads data + trains tokenizer)
3. **Verify**: Check that `~/.cache/autoresearch/` contains data shards and tokenizer

## Running

### Automated GA (recommended)

```bash
# Run the dual-lineage genetic algorithm
python dual_runner.py

# With options:
python dual_runner.py --max-generations 50 --crossover-interval 3 --seed 42

# Dry run (no actual training, random fitness values)
python dual_runner.py --dry-run
```

### Manual single experiment (same as autoresearch)

```bash
uv run train.py
```

### Agent-driven (hybrid mode)

An AI agent can also drive the loop. Point your agent here and it can:
- Use the GA framework to generate candidate genomes
- Or manually edit train.py like in the original autoresearch
- The GA provides structure; the agent provides intuition

## What You CAN Do

- Modify `train.py` — architecture, optimizer, hyperparameters, training loop
- Modify `genome.py` — add new genes, change bounds, redefine functional blocks
- Modify `genetic_algorithm.py` — change crossover/mutation/selection strategies
- Modify `dual_runner.py` — change the orchestration logic

## What You CANNOT Do

- Modify `prepare.py` — fixed evaluation, data loading, tokenizer
- Install new packages beyond `pyproject.toml`
- Modify the `evaluate_bpb` function

## The Goal

**Get the lowest val_bpb** through evolutionary search. The GA explores the configuration space systematically, and the Padgett/Wood crossover enables "innovation through recombination" — testing whether a subsystem that works in one context also works in another.

## Output Format

Same as autoresearch — each training run prints:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

## Results Logging

Results are logged to `results.tsv` (tab-separated) with columns:

```
commit  val_bpb  memory_gb  status  lineage  generation  description
```

Status: `keep`, `discard`, or `crash`
Lineage: `A`, `B`, or `baseline`

## The Evolution Loop

The dual runner alternates between lineages A and B:

```
LOOP for N generations:
    1. Pick the active lineage (alternating A/B)
    2. If crossover time:
         - Pick a random functional block
         - Transplant it from the other lineage
    3. Else:
         - Mutate the active lineage (gene-level or block-level)
    4. Inject the candidate genome into train.py
    5. git commit
    6. Train for 5 minutes
    7. Evaluate val_bpb
    8. Apply selection rule (keep/adopt/discard)
    9. Log results
```

**NEVER STOP**: Once started, the loop runs until manually interrupted or max generations reached. The human may be sleeping. You are autonomous.
