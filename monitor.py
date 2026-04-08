#!/usr/bin/env python3
"""Live ASCII progress monitor for evolve.py runs."""
import csv
import os
import sys
import time
from collections import defaultdict

RESULTS = "results.tsv"
REFRESH = 10  # seconds between refreshes

def read_results():
    """Read results.tsv and return rows grouped by problem."""
    if not os.path.exists(RESULTS):
        return {}
    problems = defaultdict(list)
    with open(RESULTS) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            pid = r["pipeline_id"]
            # Problem prefix is the island tag like "I0", but we need to detect
            # which problem based on the run. We'll group by generation blocks.
            problems["current"].append(r)
    return problems

def parse_runs(rows):
    """Split rows into separate runs (detected by generation resets)."""
    runs = []
    current_run = []
    prev_gen = -1
    for r in rows:
        gen = int(r["generation"])
        if gen < prev_gen and current_run:
            runs.append(current_run)
            current_run = []
        current_run.append(r)
        prev_gen = gen
    if current_run:
        runs.append(current_run)
    return runs

def run_stats(rows):
    """Compute stats for a single run."""
    ok = [r for r in rows if r["status"] == "ok"]
    errors = [r for r in rows if r["status"] == "error"]
    scores = []
    for r in ok:
        try:
            scores.append(float(r["score"]))
        except (ValueError, KeyError):
            pass

    gens = set(int(r["generation"]) for r in rows)
    max_gen = max(gens) if gens else 0

    # Best score per generation (running best)
    best_by_gen = {}
    running_best = float("-inf")
    for r in sorted(rows, key=lambda x: (int(x["generation"]), x["pipeline_id"])):
        g = int(r["generation"])
        if r["status"] == "ok":
            try:
                s = float(r["score"])
                running_best = max(running_best, s)
            except ValueError:
                pass
        best_by_gen[g] = running_best if running_best > float("-inf") else None

    return {
        "total": len(rows),
        "ok": len(ok),
        "errors": len(errors),
        "max_gen": max_gen,
        "best_score": max(scores) if scores else None,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "best_by_gen": best_by_gen,
        "scores": scores,
    }

def ascii_chart(best_by_gen, width=60, height=16, label=""):
    """Render an ASCII chart of best score over generations."""
    if not best_by_gen:
        return "  No data yet...\n"

    gens = sorted(best_by_gen.keys())
    vals = [best_by_gen[g] for g in gens if best_by_gen[g] is not None]
    if not vals:
        return "  No valid scores yet...\n"

    min_val = min(vals)
    max_val = max(vals)
    spread = max_val - min_val if max_val > min_val else 0.001

    # Downsample generations to fit width
    if len(gens) > width:
        step = len(gens) / width
        sampled = [gens[int(i * step)] for i in range(width)]
    else:
        sampled = gens

    lines = []
    lines.append(f"  {'Best AUC over Generations':^{width}}  {label}")
    lines.append(f"  {max_val:.4f} ┤{'':─<{width}}┐")

    for row in range(height - 2, 0, -1):
        threshold = min_val + (row / (height - 1)) * spread
        y_label = min_val + (row / (height - 1)) * spread
        line = "         │"
        for g in sampled:
            v = best_by_gen.get(g)
            if v is not None and v >= threshold:
                # Check if this is approximately at this level
                v_row = int((v - min_val) / spread * (height - 1))
                if v_row == row:
                    line += "●"
                elif v_row > row:
                    line += "│"
                else:
                    line += " "
            else:
                line += " "
        line += "│"
        lines.append(line)

    lines.append(f"  {min_val:.4f} ┤{'':─<{width}}┘")
    gen_label = f"Gen 0{' ' * (width - 8)}Gen {max(gens)}"
    lines.append(f"         └{gen_label}")

    return "\n".join(lines)

def block_switch_timeline(rows, width=55):
    """Show a timeline of algorithm diversity and block switches.

    Each character represents a generation's evaluations:
    - Letter = dominant algorithm (X=XGBoost, R=RF, G=GB, E=ExtraTrees, etc.)
    - Color intensity = whether a new best was found that gen (bright = improvement)
    """
    if not rows:
        return ""

    ALGO_CHARS = {
        "XGBoost": "X", "RandomForest": "R", "GradientBoosting": "G",
        "ExtraTrees": "E", "LightGBM": "L", "AdaBoost": "A",
        "DecisionTree": "D", "KNeighbors": "K", "SVR": "S",
        "MLP": "M", "Ridge": "r", "Lasso": "l", "ElasticNet": "e",
        "OLS": "O", "Logistic": "g", "SGD": "s", "BayesianRidge": "B",
        "GAM": "~", "CoxPH": "C", "MERF": "H", "MixedLM": "m",
    }

    import re
    gen_algos = {}  # gen -> Counter of algorithms
    gen_has_improvement = set()
    best_score = None

    for r in rows:
        if r["status"] != "ok":
            continue
        try:
            gen = int(r["generation"])
            score = float(r["score"])
        except (ValueError, KeyError):
            continue

        # Extract algorithm name
        desc = r.get("description", "")
        steps = desc.split(" -> ")
        if steps:
            alg_match = re.match(r"\[?I?\d*\]?\s*(.+)", steps[-1])
            alg_str = alg_match.group(1) if alg_match else steps[-1]
            alg_name = re.match(r"(\w+)", alg_str)
            alg_name = alg_name.group(1) if alg_name else "?"
        else:
            alg_name = "?"

        if gen not in gen_algos:
            gen_algos[gen] = {}
        gen_algos[gen][alg_name] = gen_algos[gen].get(alg_name, 0) + 1

        if best_score is None or score > best_score:
            best_score = score
            gen_has_improvement.add(gen)

    if not gen_algos:
        return ""

    gens = sorted(gen_algos.keys())
    # Downsample
    if len(gens) > width:
        step = len(gens) / width
        gens = [gens[int(i * step)] for i in range(width)]

    timeline = ""
    for g in gens:
        algos = gen_algos.get(g, {})
        if not algos:
            timeline += "·"
            continue
        # Dominant algorithm
        dominant = max(algos, key=algos.get)
        ch = ALGO_CHARS.get(dominant, "?")
        if g in gen_has_improvement:
            timeline += f"\033[1;32m{ch}\033[0m"  # bright green = improvement
        elif len(algos) > 1:
            timeline += f"\033[33m{ch}\033[0m"  # yellow = diverse gen
        else:
            timeline += f"\033[2m{ch}\033[0m"  # dim = converged

    return timeline

PROBLEM_NAMES = ["email-propensity", "event-propensity", "web-propensity"]

def display(runs_data):
    """Print full dashboard."""
    # Clear screen
    print("\033[2J\033[H", end="")

    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║            🧬  AutoML Evolutionary Search — Live Monitor  🧬           ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"║  Time: {time.strftime('%H:%M:%S')}  |  Refresh: {REFRESH}s  |  Problems: {len(runs_data)}/3 started{' ' * 11}║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()

    for i, (run_rows, name) in enumerate(zip(runs_data, PROBLEM_NAMES)):
        stats = run_stats(run_rows)
        # Last run is always potentially running; earlier runs are done
        # unless they have very few evals (might have just started)
        status = "RUNNING" if i == len(runs_data) - 1 else "DONE"
        err_rate = f"{stats['errors']/stats['total']*100:.1f}%" if stats['total'] else "0%"

        print(f"  ┌─── {name.upper()} {'─' * (50 - len(name))} [{status}]")
        print(f"  │  Generation: {stats['max_gen']:>4}  |  Evals: {stats['total']:>5} (errors: {stats['errors']}, {err_rate})")

        if stats['best_score'] is not None:
            print(f"  │  Best AUC:   {stats['best_score']:.6f}  |  Mean AUC: {stats['mean_score']:.6f}")
            timeline = block_switch_timeline(run_rows)
            if timeline:
                print(f"  │  Evolution:  {timeline}")
                print(f"  │             \033[2mAlgos: X=XGBoost E=ExtraTrees R=RF G=GradBoost L=LightGBM A=Ada D=Tree K=KNN M=MLP\033[0m")
                print(f"  │             \033[2mColor: \033[0m\033[1;32mbright\033[0m\033[2m=new best  \033[0m\033[33myellow\033[0m\033[2m=diverse  \033[0m\033[2mdim=converged\033[0m")
        else:
            print(f"  │  Best AUC:   --  |  Mean AUC: --")

        # Show chart for current/most recent run
        if stats['best_by_gen'] and stats['total'] > 5:
            print(f"  │")
            chart = ascii_chart(stats['best_by_gen'], width=55, height=12, label=name)
            for line in chart.split("\n"):
                print(f"  │  {line}")

        print(f"  └{'─' * 72}")
        print()

    # Overall summary if multiple runs done
    if len(runs_data) > 1:
        print("  ── Summary ──────────────────────────────────")
        for run_rows, name in zip(runs_data, PROBLEM_NAMES):
            s = run_stats(run_rows)
            best = f"{s['best_score']:.6f}" if s['best_score'] else "--"
            print(f"    {name:<22} Best AUC: {best}  ({s['total']} evals, gen {s['max_gen']})")
        print()

def main():
    print("Starting monitor...")

    while True:
        try:
            if os.path.exists(RESULTS):
                with open(RESULTS) as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    all_rows = list(reader)

                all_runs = parse_runs(all_rows)

                # Always show the last 3 runs (the most recent batch)
                new_runs = all_runs[-3:] if len(all_runs) >= 3 else all_runs

                if not new_runs:
                    print("\033[2J\033[H", end="")
                    print("Waiting for runs to start...")
                    time.sleep(REFRESH)
                    continue

                display(new_runs)

            time.sleep(REFRESH)

        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(REFRESH)

if __name__ == "__main__":
    main()
