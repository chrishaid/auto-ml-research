"""
ML Genome: defines the search space for AutoML genetic optimization.

A genome is a dictionary of hyperparameters and architecture choices that
fully specify a training configuration. The genetic algorithm evolves these
genomes to find optimal ML configurations.

The genome maps directly to the editable hyperparameters in train.py.
"""

import copy
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Gene definitions: each gene has a name, type, range, and default
# ---------------------------------------------------------------------------

# Gene types:
#   "float"   - continuous real value
#   "int"     - discrete integer
#   "choice"  - categorical (pick from a list)
#   "logfloat"- float sampled in log-space (good for learning rates)

GENE_DEFS = {
    # --- Model architecture ---
    "depth": {
        "type": "int", "min": 2, "max": 16, "default": 8,
        "desc": "Number of transformer layers",
    },
    "aspect_ratio": {
        "type": "int", "min": 16, "max": 128, "default": 64,
        "desc": "model_dim = depth * aspect_ratio",
    },
    "head_dim": {
        "type": "choice", "choices": [32, 64, 128, 256], "default": 128,
        "desc": "Attention head dimension",
    },
    "window_pattern": {
        "type": "choice", "choices": ["L", "SL", "SSL", "SSSL", "SSSSSL"],
        "default": "SSSL",
        "desc": "Sliding window attention pattern",
    },

    # --- Optimization ---
    "total_batch_size_exp": {
        "type": "int", "min": 14, "max": 22, "default": 19,
        "desc": "Total batch size as 2^N tokens",
    },
    "embedding_lr": {
        "type": "logfloat", "min": 0.01, "max": 2.0, "default": 0.6,
        "desc": "Learning rate for token embeddings (Adam)",
    },
    "unembedding_lr": {
        "type": "logfloat", "min": 0.0005, "max": 0.05, "default": 0.004,
        "desc": "Learning rate for lm_head (Adam)",
    },
    "matrix_lr": {
        "type": "logfloat", "min": 0.005, "max": 0.2, "default": 0.04,
        "desc": "Learning rate for matrix parameters (Muon)",
    },
    "scalar_lr": {
        "type": "logfloat", "min": 0.05, "max": 2.0, "default": 0.5,
        "desc": "Learning rate for per-layer scalars (Adam)",
    },
    "weight_decay": {
        "type": "logfloat", "min": 0.01, "max": 1.0, "default": 0.2,
        "desc": "Cautious weight decay for Muon",
    },
    "adam_beta1": {
        "type": "float", "min": 0.5, "max": 0.99, "default": 0.8,
        "desc": "Adam beta1",
    },
    "adam_beta2": {
        "type": "float", "min": 0.9, "max": 0.999, "default": 0.95,
        "desc": "Adam beta2",
    },
    "warmup_ratio": {
        "type": "float", "min": 0.0, "max": 0.3, "default": 0.0,
        "desc": "Fraction of time budget for LR warmup",
    },
    "warmdown_ratio": {
        "type": "float", "min": 0.1, "max": 0.8, "default": 0.5,
        "desc": "Fraction of time budget for LR warmdown",
    },
    "final_lr_frac": {
        "type": "float", "min": 0.0, "max": 0.5, "default": 0.0,
        "desc": "Final LR as fraction of initial",
    },

    # --- Training ---
    "device_batch_size": {
        "type": "choice", "choices": [16, 32, 64, 128, 256], "default": 128,
        "desc": "Per-device batch size",
    },

    # --- Model tweaks ---
    "softcap": {
        "type": "float", "min": 5.0, "max": 50.0, "default": 15.0,
        "desc": "Logit softcap value (tanh scaling)",
    },
    "mlp_expansion": {
        "type": "choice", "choices": [2, 3, 4, 6, 8], "default": 4,
        "desc": "MLP hidden dim expansion factor",
    },
    "activation": {
        "type": "choice", "choices": ["relu_sq", "gelu", "swiglu"], "default": "relu_sq",
        "desc": "MLP activation function",
    },
    "use_value_embeds": {
        "type": "choice", "choices": [True, False], "default": True,
        "desc": "Whether to use ResFormer-style value embeddings",
    },
    "muon_ns_steps": {
        "type": "int", "min": 1, "max": 5, "default": 5,
        "desc": "Muon polar orthogonalization Newton-Schulz steps",
    },
    "x0_lambda_init": {
        "type": "logfloat", "min": 0.01, "max": 1.0, "default": 0.1,
        "desc": "Initial value for skip-connection lambdas",
    },
}


# ---------------------------------------------------------------------------
# Genome class
# ---------------------------------------------------------------------------

class Genome:
    """
    A complete ML configuration represented as a dictionary of genes.

    Each gene corresponds to a tunable hyperparameter or architecture choice
    in train.py. The genome can be mutated, crossed over with another genome,
    and serialized/deserialized.
    """

    def __init__(self, genes: Optional[dict] = None):
        if genes is not None:
            self.genes = dict(genes)
        else:
            # Initialize with defaults
            self.genes = {name: gdef["default"] for name, gdef in GENE_DEFS.items()}

    @classmethod
    def random(cls, rng: Optional[random.Random] = None) -> "Genome":
        """Create a random genome with all genes sampled from their ranges."""
        r = rng or random.Random()
        genes = {}
        for name, gdef in GENE_DEFS.items():
            genes[name] = _random_gene(name, gdef, r)
        return cls(genes)

    @classmethod
    def from_defaults(cls) -> "Genome":
        """Create a genome with the baseline default values."""
        return cls()

    def copy(self) -> "Genome":
        return Genome(copy.deepcopy(self.genes))

    def clamp(self) -> "Genome":
        """Clamp all genes to their valid ranges."""
        for name, gdef in GENE_DEFS.items():
            if name not in self.genes:
                self.genes[name] = gdef["default"]
                continue
            val = self.genes[name]
            if gdef["type"] in ("float", "logfloat"):
                self.genes[name] = max(gdef["min"], min(gdef["max"], float(val)))
            elif gdef["type"] == "int":
                self.genes[name] = max(gdef["min"], min(gdef["max"], int(round(val))))
            elif gdef["type"] == "choice":
                if val not in gdef["choices"]:
                    self.genes[name] = gdef["default"]
        return self

    def diff(self, other: "Genome") -> dict:
        """Return genes that differ between self and other."""
        diffs = {}
        for name in GENE_DEFS:
            if self.genes.get(name) != other.genes.get(name):
                diffs[name] = (self.genes.get(name), other.genes.get(name))
        return diffs

    def to_dict(self) -> dict:
        return dict(self.genes)

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(genes=d)

    def to_json(self) -> str:
        return json.dumps(self.genes, indent=2, default=str)

    @classmethod
    def from_json(cls, s: str) -> "Genome":
        return cls(genes=json.loads(s))

    def save(self, path: str):
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str) -> "Genome":
        return cls.from_json(Path(path).read_text())

    # -------------------------------------------------------------------
    # Generate the hyperparameters section for train.py
    # -------------------------------------------------------------------

    def to_train_config(self) -> str:
        """
        Generate the Python code for the hyperparameters section of train.py.

        Returns a string that can be spliced into train.py between the
        hyperparameters markers.
        """
        g = self.genes
        lines = [
            '# --- BEGIN GENOME-GENERATED HYPERPARAMETERS ---',
            f'ASPECT_RATIO = {g["aspect_ratio"]}',
            f'HEAD_DIM = {g["head_dim"]}',
            f'WINDOW_PATTERN = "{g["window_pattern"]}"',
            '',
            f'TOTAL_BATCH_SIZE = 2**{g["total_batch_size_exp"]}',
            f'EMBEDDING_LR = {g["embedding_lr"]}',
            f'UNEMBEDDING_LR = {g["unembedding_lr"]}',
            f'MATRIX_LR = {g["matrix_lr"]}',
            f'SCALAR_LR = {g["scalar_lr"]}',
            f'WEIGHT_DECAY = {g["weight_decay"]}',
            f'ADAM_BETAS = ({g["adam_beta1"]}, {g["adam_beta2"]})',
            f'WARMUP_RATIO = {g["warmup_ratio"]}',
            f'WARMDOWN_RATIO = {g["warmdown_ratio"]}',
            f'FINAL_LR_FRAC = {g["final_lr_frac"]}',
            '',
            f'DEPTH = {g["depth"]}',
            f'DEVICE_BATCH_SIZE = {g["device_batch_size"]}',
            '# --- END GENOME-GENERATED HYPERPARAMETERS ---',
        ]
        return '\n'.join(lines)

    def describe(self) -> str:
        """One-line description of key parameters."""
        g = self.genes
        batch_exp = g["total_batch_size_exp"]
        return (
            f"d={g['depth']} ar={g['aspect_ratio']} hd={g['head_dim']} "
            f"bs=2^{batch_exp} mlr={g['matrix_lr']:.4f} elr={g['embedding_lr']:.3f} "
            f"wd={g['weight_decay']:.3f} act={g['activation']} "
            f"wp={g['window_pattern']}"
        )

    def __repr__(self):
        return f"Genome({self.describe()})"


# ---------------------------------------------------------------------------
# Gene manipulation helpers
# ---------------------------------------------------------------------------

def _random_gene(name: str, gdef: dict, rng: random.Random):
    """Sample a random value for a gene."""
    gtype = gdef["type"]
    if gtype == "float":
        return rng.uniform(gdef["min"], gdef["max"])
    elif gtype == "logfloat":
        log_min = math.log(gdef["min"])
        log_max = math.log(gdef["max"])
        return math.exp(rng.uniform(log_min, log_max))
    elif gtype == "int":
        return rng.randint(gdef["min"], gdef["max"])
    elif gtype == "choice":
        return rng.choice(gdef["choices"])
    raise ValueError(f"Unknown gene type: {gtype}")


def mutate_gene(name: str, current_val, rng: random.Random, strength: float = 0.3):
    """
    Mutate a single gene. Returns the new value.

    strength: 0-1, controls how far from current value the mutation can go.
    """
    gdef = GENE_DEFS[name]
    gtype = gdef["type"]

    if gtype == "float":
        span = (gdef["max"] - gdef["min"]) * strength
        new_val = current_val + rng.gauss(0, span * 0.5)
        return max(gdef["min"], min(gdef["max"], new_val))

    elif gtype == "logfloat":
        log_val = math.log(max(current_val, 1e-10))
        log_span = (math.log(gdef["max"]) - math.log(gdef["min"])) * strength
        new_log = log_val + rng.gauss(0, log_span * 0.5)
        new_log = max(math.log(gdef["min"]), min(math.log(gdef["max"]), new_log))
        return math.exp(new_log)

    elif gtype == "int":
        span = (gdef["max"] - gdef["min"]) * strength
        delta = int(round(rng.gauss(0, max(1, span * 0.5))))
        new_val = current_val + delta
        return max(gdef["min"], min(gdef["max"], new_val))

    elif gtype == "choice":
        # With probability proportional to strength, pick a random choice
        if rng.random() < strength:
            return rng.choice(gdef["choices"])
        return current_val

    return current_val


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = random.Random(42)

    print("=== ML Genome Test ===\n")

    # Default genome
    default = Genome.from_defaults()
    print(f"Default: {default}")
    print(f"Config preview:\n{default.to_train_config()}\n")

    # Random genome
    rand = Genome.random(rng)
    print(f"Random:  {rand}")

    # Diff
    diffs = default.diff(rand)
    print(f"\nDifferences ({len(diffs)} genes):")
    for name, (v1, v2) in sorted(diffs.items()):
        print(f"  {name}: {v1} -> {v2}")

    # Mutation
    mutated = default.copy()
    for name in mutated.genes:
        mutated.genes[name] = mutate_gene(name, mutated.genes[name], rng, strength=0.3)
    mutated.clamp()
    print(f"\nMutated: {mutated}")

    # Serialization roundtrip
    json_str = default.to_json()
    loaded = Genome.from_json(json_str)
    assert loaded.genes == default.genes, "JSON roundtrip failed"
    print("\nJSON roundtrip: OK")
