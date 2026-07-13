"""Generate the three benchmark plots from `results/benchmark.csv` + `histories.pkl`.

Plot 1  cond(H(x*))  vs iterations               (log-log)
Plot 2  n            vs runtime                  (log-log)
Plot 3  f(x_k) - f*  vs k for 3 chosen problems  (semi/log-log)
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from collections import defaultdict
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


SOLVERS = ("GD", "Newton", "BFGS")
MARKERS = {"GD": "o", "Newton": "s", "BFGS": "^"}
COLORS = {"GD": "tab:blue", "Newton": "tab:orange", "BFGS": "tab:green"}


def load_records(csv_path: str) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row["n"] = int(row["n"])
            row["iterations"] = int(row["iterations"])
            row["runtime_s"] = float(row["runtime_s"])
            row["f_final"] = float(row["f_final"])
            row["cond_hessian"] = float(row["cond_hessian"])
            row["grad_norm"] = float(row["grad_norm"])
            row["converged"] = row["converged"] == "True"
            rows.append(row)
    return rows


def _finite_positive(pairs: Iterable[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for x, y in pairs:
        if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0:
            xs.append(x)
            ys.append(y)
    return np.asarray(xs), np.asarray(ys)


def plot_cond_vs_iters(records: list[dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    per_solver = defaultdict(list)
    for r in records:
        if r["iterations"] > 0:
            per_solver[r["solver"]].append((r["cond_hessian"], r["iterations"]))

    for solver in SOLVERS:
        xs, ys = _finite_positive(per_solver.get(solver, []))
        if xs.size == 0:
            continue
        ax.scatter(xs, ys, marker=MARKERS[solver], s=70,
                   label=solver, color=COLORS[solver],
                   edgecolors="black", linewidths=0.5, alpha=0.85)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Condition number  $\kappa(\nabla^2 f(x^*))$")
    ax.set_ylabel("Iterations to convergence")
    ax.set_title("Iterations vs Hessian conditioning at the solution")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_n_vs_runtime(records: list[dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    per_solver = defaultdict(list)
    for r in records:
        if r["iterations"] > 0:
            per_solver[r["solver"]].append((r["n"], r["runtime_s"]))

    for solver in SOLVERS:
        xs, ys = _finite_positive(per_solver.get(solver, []))
        if xs.size == 0:
            continue
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], marker=MARKERS[solver],
                label=solver, color=COLORS[solver],
                linestyle="", markersize=8,
                markeredgecolor="black", markeredgewidth=0.5, alpha=0.85)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of variables  $n$")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime vs problem size")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def _pick_three(records: list[dict], histories: dict) -> list[str]:
    """Pick 3 representative problems: small n, medium n, largest n (all with
    at least one solver that produced a usable history)."""
    per_problem_n = {}
    for r in records:
        per_problem_n[r["problem"]] = r["n"]

    usable = []
    for name in per_problem_n:
        h = histories.get(name, {}).get("solvers", {})
        if any("f_history" in v and len(v["f_history"]) > 1 for v in h.values()):
            usable.append(name)
    if not usable:
        return []

    usable_sorted = sorted(usable, key=lambda p: per_problem_n[p])
    if len(usable_sorted) <= 3:
        return usable_sorted
    return [usable_sorted[0], usable_sorted[len(usable_sorted) // 2], usable_sorted[-1]]


def plot_convergence_histories(
    records: list[dict],
    histories: dict,
    out_path: str,
    chosen: list[str] | None = None,
) -> None:
    picks = chosen or _pick_three(records, histories)
    if not picks:
        print("no usable histories to plot")
        return
    picks = picks[:3]

    fig, axes = plt.subplots(1, len(picks), figsize=(6.0 * len(picks), 5.0), squeeze=False)
    axes = axes[0]

    for ax, name in zip(axes, picks):
        entry = histories.get(name, {})
        n = entry.get("n", "?")
        solvers = entry.get("solvers", {})

        # Reference f*: minimum final f across solvers on this problem.
        f_stars = []
        for solver in SOLVERS:
            info = solvers.get(solver, {})
            if "f_history" in info and len(info["f_history"]):
                f_stars.append(info["f_history"][-1])
        f_star = min(f_stars) if f_stars else 0.0

        cond_lines = []
        for solver in SOLVERS:
            info = solvers.get(solver, {})
            hist = info.get("f_history")
            if not hist or len(hist) < 2:
                continue
            k = np.arange(len(hist))
            gap = np.asarray(hist, dtype=float) - f_star
            # For log-y plotting we need strictly positive values.
            eps = 1e-16
            gap = np.maximum(gap, eps)
            ax.plot(k, gap, color=COLORS[solver], marker=MARKERS[solver],
                    markersize=4, markevery=max(1, len(hist) // 20),
                    label=solver, linewidth=1.5)
            cond_val = info.get("cond_hessian", float("nan"))
            cond_lines.append(f"{solver}: $\\kappa$={cond_val:.2e}")

        ax.set_yscale("log")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel(r"$f(x_k) - f^*$")
        title = f"{name}  (n = {n})"
        ax.set_title(title)
        # Underline dimension + per-solver cond in a text box below the legend.
        leg = ax.legend(frameon=True, loc="upper right")
        cond_text = "\n".join(cond_lines) if cond_lines else ""
        if cond_text:
            ax.text(0.02, 0.02, cond_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))
        ax.grid(True, which="both", ls=":", alpha=0.6)

    fig.suptitle("Convergence history: $f(x_k) - f^*$ vs iterations", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--convergence-problems", nargs="+", default=None,
                        help="Override the 3 problems for Plot 3.")
    args = parser.parse_args(argv)

    os.makedirs(args.figures_dir, exist_ok=True)

    csv_path = os.path.join(args.results_dir, "benchmark.csv")
    pkl_path = os.path.join(args.results_dir, "histories.pkl")

    records = load_records(csv_path)
    with open(pkl_path, "rb") as fh:
        histories = pickle.load(fh)

    plot_cond_vs_iters(
        records, os.path.join(args.figures_dir, "plot1_cond_vs_iters.png")
    )
    plot_n_vs_runtime(
        records, os.path.join(args.figures_dir, "plot2_n_vs_runtime.png")
    )
    plot_convergence_histories(
        records, histories,
        os.path.join(args.figures_dir, "plot3_convergence.png"),
        chosen=args.convergence_problems,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
