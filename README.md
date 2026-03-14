# automl-evolve

Genetic AutoML with dual-lineage evolution and Padgett/Wood-style block crossover. Forked from [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy.

## What is this?

Two model configurations evolve **independently** on the same GPU, each running the autoresearch loop: mutate the config, train for 5 minutes, check val_bpb, keep or discard. Every few generations, we do something interesting — we **swap a functional block** between the two lineages:

```python
# Two lineages each have a genome (a set of ML hyperparameters)
lineage_A = Genome(depth=8, matrix_lr=0.04, activation="relu_sq", ...)
lineage_B = Genome(depth=6, matrix_lr=0.02, activation="gelu", ...)

# Padgett/Wood crossover: transplant a functional block
child = lineage_A.copy()
child.learning_rates = lineage_B.learning_rates   # swap the whole LR subsystem

# Train and evaluate
child_bpb = train_5min(child)

# Three-way decision
if child_bpb < lineage_A.bpb:    keep as new A
elif child_bpb < lineage_B.bpb:  B adopts child
else:                              discard
```

This is inspired by Padgett & Wood's research on organizational learning through "code crossover" — innovation comes from recombining *functional routines* between entities, not from gradual parameter tuning.

## How it works

The system has four layers:

| File | Role |
|------|------|
| `prepare.py` | Fixed data prep, tokenizer, evaluation (from autoresearch, **do not modify**) |
| `train.py` | Training script — the GA injects hyperparameters here |
| `genome.py` | Defines the search space: genes, bounds, functional blocks |
| `genetic_algorithm.py` | Mutation, block crossover, selection logic |
| `dual_runner.py` | **Main entry point** — orchestrates two parallel lineages |

### Functional blocks

Crossover doesn't swap random genes — it swaps *coherent subsystems*:

- **attention**: head_dim, window_pattern, value embeddings
- **model_scale**: depth, aspect_ratio, batch sizes
- **learning_rates**: LR for embeddings, matrices, scalars
- **optimizer_dynamics**: weight decay, Adam betas, Muon steps
- **lr_schedule**: warmup, warmdown, final LR
- **mlp**: expansion factor, activation function
- **residual**: softcap, skip-connection scaling

## Quick start

```bash
# Install
uv sync

# Prepare data (one-time)
uv run prepare.py

# Run the genetic algorithm
python dual_runner.py

# With options
python dual_runner.py --max-generations 50 --crossover-interval 3

# Dry run (no training, random fitness)
python dual_runner.py --dry-run
```

## Project structure

```
prepare.py              — constants, data prep, evaluation (do not modify)
train.py                — model + training loop (GA modifies hyperparams here)
genome.py               — gene definitions, Genome class, mutation helpers
genetic_algorithm.py    — crossover, selection, lineage tracking
dual_runner.py          — dual-lineage orchestrator (main entry point)
program.md              — agent instructions (detailed workflow)
```

## Design

- **Dual lineages**: Two configurations evolve independently, giving diversity. Neither can get permanently stuck — the other lineage may rescue it via crossover.
- **Block crossover (Padgett/Wood)**: Swap coherent subsystems, not random parameters. This preserves internal consistency within each block while testing cross-context transfer.
- **Three-way selection**: A crossover child can be adopted by *either* parent, not just the recipient. This doubles the chance of a useful innovation surviving.
- **Fixed 5-minute budget**: Every experiment is fairly comparable regardless of configuration. ~12 experiments/hour, ~100 overnight.
- **Git-based**: Every experiment is a commit. Full reproducibility, easy rollback.

## References

- [autoresearch](https://github.com/karpathy/autoresearch) — the original autonomous LLM research framework
- Padgett, J.F. & Powell, W.W. (2012). *The Emergence of Organizations and Markets*. Princeton University Press. — The organizational learning theory behind block crossover.

## License

MIT
