"""
Genetic Algorithm with Padgett/Wood-style Block Crossover.

Implements the core GA operations for evolving ML configurations:
- Mutation: random perturbation of individual genes
- Block crossover: swap functional blocks between two parent configurations
  (inspired by Padgett & Wood's organizational learning through code crossover)
- Selection: keep improvements, discard regressions

The key insight from Padgett & Wood (2012) is that innovation often comes from
recombining *functional routines* between organizations, not from gradual
parameter tuning. Here, a "routine" is a coherent block of ML configuration:
the attention setup, the optimizer config, the MLP architecture, etc.

Two model lineages evolve independently, and periodically a functional block
from one is transplanted into the other. If the hybrid improves on either
parent, it survives. Otherwise, revert.
"""

import copy
import random
from dataclasses import dataclass, field
from typing import Optional

from genome import Genome, GENE_DEFS, mutate_gene, _random_gene


# ---------------------------------------------------------------------------
# Functional blocks: groups of genes that form coherent "routines"
# ---------------------------------------------------------------------------

# Each block is a named group of genes that get swapped together during
# crossover. This mirrors Padgett & Wood's concept of swapping functional
# routines - you don't swap a single parameter, you swap an entire
# coherent subsystem.

FUNCTIONAL_BLOCKS = {
    "attention": {
        "genes": ["head_dim", "window_pattern", "use_value_embeds"],
        "desc": "Attention mechanism configuration",
    },
    "model_scale": {
        "genes": ["depth", "aspect_ratio", "device_batch_size", "total_batch_size_exp"],
        "desc": "Model size and batch configuration",
    },
    "learning_rates": {
        "genes": ["embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr"],
        "desc": "Learning rate configuration across parameter groups",
    },
    "optimizer_dynamics": {
        "genes": ["weight_decay", "adam_beta1", "adam_beta2", "muon_ns_steps"],
        "desc": "Optimizer momentum and regularization",
    },
    "lr_schedule": {
        "genes": ["warmup_ratio", "warmdown_ratio", "final_lr_frac"],
        "desc": "Learning rate schedule shape",
    },
    "mlp": {
        "genes": ["mlp_expansion", "activation"],
        "desc": "MLP architecture (expansion factor + activation)",
    },
    "residual": {
        "genes": ["softcap", "x0_lambda_init"],
        "desc": "Residual connection and output scaling",
    },
}

# Flat list of all block names for random selection
BLOCK_NAMES = list(FUNCTIONAL_BLOCKS.keys())


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate(genome: Genome,
           mutation_rate: float = 0.3,
           mutation_strength: float = 0.3,
           rng: Optional[random.Random] = None) -> Genome:
    """
    Mutate a genome by perturbing individual genes.

    Args:
        genome: The parent genome to mutate.
        mutation_rate: Probability that each gene is mutated (0-1).
        mutation_strength: How far mutations can go from current value (0-1).
        rng: Random number generator for reproducibility.

    Returns:
        A new mutated genome (parent is not modified).
    """
    r = rng or random.Random()
    child = genome.copy()

    for name in GENE_DEFS:
        if r.random() < mutation_rate:
            child.genes[name] = mutate_gene(name, child.genes[name], r, mutation_strength)

    return child.clamp()


def mutate_block(genome: Genome,
                 block_name: Optional[str] = None,
                 strength: float = 0.5,
                 rng: Optional[random.Random] = None) -> Genome:
    """
    Mutate all genes within a single functional block.

    This is a "focused mutation" - rather than scattershot changes across
    the whole genome, it intensively explores one subsystem at a time.
    """
    r = rng or random.Random()
    if block_name is None:
        block_name = r.choice(BLOCK_NAMES)

    child = genome.copy()
    block = FUNCTIONAL_BLOCKS[block_name]

    for gene_name in block["genes"]:
        child.genes[gene_name] = mutate_gene(gene_name, child.genes[gene_name], r, strength)

    return child.clamp()


# ---------------------------------------------------------------------------
# Padgett/Wood Block Crossover
# ---------------------------------------------------------------------------

@dataclass
class CrossoverResult:
    """Result of a block crossover operation."""
    child: Genome
    donor_lineage: str           # which lineage donated the block
    recipient_lineage: str       # which lineage received the block
    block_name: str              # which functional block was swapped
    block_genes: dict            # the actual gene values that were transplanted
    description: str             # human-readable description

    def __repr__(self):
        return (f"CrossoverResult(block={self.block_name}, "
                f"{self.donor_lineage}->{self.recipient_lineage}, "
                f"genes={list(self.block_genes.keys())})")


def block_crossover(recipient: Genome,
                    donor: Genome,
                    block_name: Optional[str] = None,
                    rng: Optional[random.Random] = None) -> CrossoverResult:
    """
    Padgett/Wood-style block crossover: transplant a functional block
    from the donor into the recipient.

    This is NOT uniform crossover (random gene-by-gene mixing). Instead,
    it swaps a coherent *functional routine* - a group of genes that work
    together as a subsystem. This preserves the internal coherence of
    each block while testing whether a subsystem that works well in one
    context also works in another.

    Args:
        recipient: The genome receiving the transplanted block.
        donor: The genome donating the block.
        block_name: Which block to swap. If None, chosen randomly.
        rng: Random number generator.

    Returns:
        CrossoverResult with the child genome and metadata.
    """
    r = rng or random.Random()
    if block_name is None:
        block_name = r.choice(BLOCK_NAMES)

    block = FUNCTIONAL_BLOCKS[block_name]
    child = recipient.copy()

    # Transplant all genes in the block from donor to recipient
    transplanted = {}
    for gene_name in block["genes"]:
        old_val = child.genes.get(gene_name)
        new_val = donor.genes.get(gene_name)
        if new_val is not None:
            child.genes[gene_name] = copy.deepcopy(new_val)
            transplanted[gene_name] = new_val

    child.clamp()

    desc = (f"Crossover: transplanted '{block_name}' block "
            f"({block['desc']}) from donor")

    return CrossoverResult(
        child=child,
        donor_lineage="",   # filled in by the runner
        recipient_lineage="",
        block_name=block_name,
        block_genes=transplanted,
        description=desc,
    )


def multi_block_crossover(recipient: Genome,
                          donor: Genome,
                          num_blocks: int = 2,
                          rng: Optional[random.Random] = None) -> CrossoverResult:
    """
    Swap multiple functional blocks at once.

    More aggressive than single-block crossover - useful when the two
    lineages have diverged significantly and you want to test bigger
    architectural leaps.
    """
    r = rng or random.Random()
    blocks = r.sample(BLOCK_NAMES, min(num_blocks, len(BLOCK_NAMES)))

    child = recipient.copy()
    all_transplanted = {}

    for block_name in blocks:
        block = FUNCTIONAL_BLOCKS[block_name]
        for gene_name in block["genes"]:
            new_val = donor.genes.get(gene_name)
            if new_val is not None:
                child.genes[gene_name] = copy.deepcopy(new_val)
                all_transplanted[gene_name] = new_val

    child.clamp()

    block_names_str = "+".join(blocks)
    desc = f"Multi-crossover: transplanted [{block_names_str}] from donor"

    return CrossoverResult(
        child=child,
        donor_lineage="",
        recipient_lineage="",
        block_name=block_names_str,
        block_genes=all_transplanted,
        description=desc,
    )


# ---------------------------------------------------------------------------
# Selection logic (Padgett/Wood decision rule)
# ---------------------------------------------------------------------------

@dataclass
class SelectionDecision:
    """The outcome of evaluating a crossover child."""
    action: str              # "keep_as_recipient", "adopt_as_donor", "discard"
    child: Genome
    child_fitness: float
    recipient_fitness: float
    donor_fitness: float
    description: str


def select(child_fitness: float,
           recipient_fitness: float,
           donor_fitness: float,
           child: Genome,
           improvement_threshold: float = 0.0) -> SelectionDecision:
    """
    Padgett/Wood selection rule:

    1. If child beats recipient -> recipient adopts child (primary improvement)
    2. Elif child beats donor -> donor adopts child (the swap helped the other side)
    3. Else -> discard (the transplant didn't help anyone)

    For val_bpb, LOWER is better, so "beats" means child < parent.

    Args:
        child_fitness: The child's val_bpb (lower = better).
        recipient_fitness: The recipient parent's val_bpb.
        donor_fitness: The donor parent's val_bpb.
        child: The child genome.
        improvement_threshold: Minimum improvement to count (avoids noise).

    Returns:
        SelectionDecision describing what to do.
    """
    # For val_bpb: lower is better, so improvement = parent - child > threshold
    improves_recipient = (recipient_fitness - child_fitness) > improvement_threshold
    improves_donor = (donor_fitness - child_fitness) > improvement_threshold

    if improves_recipient:
        return SelectionDecision(
            action="keep_as_recipient",
            child=child,
            child_fitness=child_fitness,
            recipient_fitness=recipient_fitness,
            donor_fitness=donor_fitness,
            description=(f"Child ({child_fitness:.6f}) beats recipient "
                        f"({recipient_fitness:.6f}) by "
                        f"{recipient_fitness - child_fitness:.6f}"),
        )
    elif improves_donor:
        return SelectionDecision(
            action="adopt_as_donor",
            child=child,
            child_fitness=child_fitness,
            recipient_fitness=recipient_fitness,
            donor_fitness=donor_fitness,
            description=(f"Child ({child_fitness:.6f}) beats donor "
                        f"({donor_fitness:.6f}) by "
                        f"{donor_fitness - child_fitness:.6f}"),
        )
    else:
        return SelectionDecision(
            action="discard",
            child=child,
            child_fitness=child_fitness,
            recipient_fitness=recipient_fitness,
            donor_fitness=donor_fitness,
            description=(f"Child ({child_fitness:.6f}) doesn't beat "
                        f"recipient ({recipient_fitness:.6f}) or "
                        f"donor ({donor_fitness:.6f})"),
        )


# ---------------------------------------------------------------------------
# Population-level helpers
# ---------------------------------------------------------------------------

@dataclass
class Lineage:
    """
    One evolutionary lineage: a current best genome + its fitness history.

    Each lineage evolves independently via mutation, and periodically
    participates in crossover with the other lineage.
    """
    name: str
    genome: Genome
    fitness: float = float("inf")  # val_bpb (lower = better), inf = not yet evaluated
    generation: int = 0
    history: list = field(default_factory=list)  # list of (generation, fitness, description)

    def record(self, fitness: float, description: str):
        """Record a generation's result."""
        self.history.append({
            "generation": self.generation,
            "fitness": fitness,
            "description": description,
        })

    def advance(self, new_genome: Genome, new_fitness: float, description: str):
        """Accept a new genome as the lineage's current best."""
        self.genome = new_genome
        self.fitness = new_fitness
        self.generation += 1
        self.record(new_fitness, f"KEEP: {description}")

    def reject(self, fitness: float, description: str):
        """Record a rejected experiment."""
        self.generation += 1
        self.record(fitness, f"DISCARD: {description}")

    def crash(self, description: str):
        """Record a crashed experiment."""
        self.generation += 1
        self.record(0.0, f"CRASH: {description}")


def should_crossover(generation: int,
                     crossover_interval: int = 5,
                     rng: Optional[random.Random] = None) -> bool:
    """
    Decide whether to attempt crossover at this generation.

    Crossover happens every `crossover_interval` generations, with some
    randomness to avoid strict periodicity.
    """
    r = rng or random.Random()
    if generation < 2:
        return False  # let each lineage establish a baseline first
    if generation % crossover_interval == 0:
        return True
    # Small random chance even off-cycle (Padgett/Wood: chance encounters)
    return r.random() < 0.1


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = random.Random(42)

    print("=== Genetic Algorithm Test ===\n")

    # Create two lineages with different genomes
    a = Genome.from_defaults()
    b = Genome.random(rng)

    print(f"Lineage A: {a}")
    print(f"Lineage B: {b}")

    # Mutation
    a_mut = mutate(a, mutation_rate=0.3, mutation_strength=0.3, rng=rng)
    print(f"\nA mutated: {a_mut}")
    print(f"  Diffs: {len(a.diff(a_mut))} genes changed")

    # Block mutation
    a_block = mutate_block(a, "learning_rates", strength=0.5, rng=rng)
    print(f"\nA block-mutated (learning_rates): {a_block}")

    # Block crossover
    result = block_crossover(a, b, rng=rng)
    print(f"\nCrossover: {result}")
    print(f"  {result.description}")

    # Multi-block crossover
    result2 = multi_block_crossover(a, b, num_blocks=2, rng=rng)
    print(f"\nMulti-crossover: {result2}")
    print(f"  {result2.description}")

    # Selection
    decision = select(0.95, 0.97, 0.96, result.child)
    print(f"\nSelection: {decision.action}")
    print(f"  {decision.description}")

    # Functional blocks summary
    print(f"\n--- Functional Blocks ---")
    for name, block in FUNCTIONAL_BLOCKS.items():
        print(f"  {name:20s}: {block['genes']}")
        print(f"  {'':20s}  {block['desc']}")
