"""Benchmark GD / Newton / BFGS on CUTEst problems.

Filters CUTEst for the official set (objective='other', constraints='unconstrained',
regular=True), runs each solver on every instance, and records:
    problem, n, iterations, runtime (s), f(x*), cond(H(x*)).

Results are written to `results/benchmark.csv`, and the full per-iteration
`f_history` for each (problem, solver) is pickled to `results/histories.pkl`
so that `plots.py` can build the convergence-history plot without re-running
the benchmark.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np
import pycutest

from methods import gradient_descent, newton, bfgs


BENCHMARK_PROBLEMS: list[str] = [
    "ROSENBR", "ALLINITU", "BEALE", "BOOTH", "BOX3",
    "DENSCHNB", "DIXMAANA1", "ENGVAL1", "FREUROTH", "HUMPS",
    "MARATOSB", "NONDQUAR", "SCHMVETT", "TRIDIA", "ZANGWIL3",
]


@dataclass
class Record:
    problem: str
    solver: str
    n: int
    iterations: int
    runtime_s: float
    f_final: float
    grad_norm: float
    cond_hessian: float
    converged: bool


def make_callbacks(problem):
    """Wrap pycutest into (obj_grad, hess) callables."""
    def obj_grad(x: np.ndarray):
        f, g = problem.obj(x, gradient=True)
        return float(f), np.asarray(g, dtype=float)

    def hess(x: np.ndarray):
        H = problem.hess(x)
        return np.asarray(H, dtype=float)

    return obj_grad, hess


def cond_at(x: np.ndarray, hess: Callable[[np.ndarray], np.ndarray]) -> float:
    """Condition number of the exact Hessian at x. Robust to failures."""
    try:
        H = hess(x)
        if not np.all(np.isfinite(H)):
            return float("nan")
        return float(np.linalg.cond(H))
    except Exception:
        return float("nan")


def run_one(name: str, max_iter_gd: int, max_iter_newton: int, max_iter_bfgs: int
            ) -> tuple[list[Record], dict]:
    """Run all three solvers on one problem. Returns (records, histories)."""
    problem = pycutest.import_problem(name)
    x0 = np.asarray(problem.x0, dtype=float).copy()
    n = int(problem.n)
    obj_grad, hess = make_callbacks(problem)

    records: list[Record] = []
    histories: dict = {"n": n, "solvers": {}}

    runs = [
        ("GD",     lambda: gradient_descent(obj_grad, x0.copy(), max_iter=max_iter_gd)),
        ("Newton", lambda: newton(obj_grad, hess, x0.copy(), max_iter=max_iter_newton)),
        ("BFGS",   lambda: bfgs(obj_grad, x0.copy(), max_iter=max_iter_bfgs)),
    ]

    for solver_name, run in runs:
        t0 = time.perf_counter()
        try:
            result = run()
            elapsed = time.perf_counter() - t0
            cnum = cond_at(result["x"], hess)
            rec = Record(
                problem=name,
                solver=solver_name,
                n=n,
                iterations=int(result["iterations"]),
                runtime_s=float(elapsed),
                f_final=float(result["f"]),
                grad_norm=float(result["grad_norm"]),
                cond_hessian=float(cnum),
                converged=bool(result["converged"]),
            )
            histories["solvers"][solver_name] = {
                "f_history": list(result["f_history"]),
                "grad_history": list(result["grad_history"]),
                "cond_hessian": float(cnum),
                "iterations": int(result["iterations"]),
                "converged": bool(result["converged"]),
            }
        except Exception as e:  # log failures without killing the benchmark
            elapsed = time.perf_counter() - t0
            rec = Record(
                problem=name, solver=solver_name, n=n,
                iterations=-1, runtime_s=float(elapsed),
                f_final=float("nan"), grad_norm=float("nan"),
                cond_hessian=float("nan"), converged=False,
            )
            histories["solvers"][solver_name] = {"error": repr(e)}
        records.append(rec)
        print(
            f"  {solver_name:6s}  iters={rec.iterations:5d}  "
            f"t={rec.runtime_s:7.3f}s  f={rec.f_final:.6e}  "
            f"|g|={rec.grad_norm:.2e}  cond={rec.cond_hessian:.3e}  "
            f"conv={rec.converged}"
        )
    return records, histories


def list_problems(limit: Optional[int]) -> list[str]:
    names = sorted(pycutest.find_problems(
        objective="other",
        constraints="unconstrained",
        regular=True,
    ))
    if limit is not None:
        names = names[:limit]
    return names


def filter_available(names: list[str]) -> list[str]:
    """Drop names not present in the local CUTEst database.

    Some malformed SIF entries abort at Fortran runtime (not a catchable
    Python exception), which would kill the whole benchmark. Screening
    against the master problem list up-front keeps the run safe.
    """
    available = set(pycutest.find_problems())
    kept, missing = [], []
    for n in names:
        (kept if n in available else missing).append(n)
    if missing:
        print(f"!! Skipping {len(missing)} problem(s) not in local CUTEst: {missing}")
    return kept


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on number of problems (default: all matches).")
    parser.add_argument("--problems", nargs="+", default=None,
                        help="Explicit problem names (overrides the filter).")
    parser.add_argument("--all", action="store_true",
                        help="Run on all matching MASTSIF unconstrained problems "
                             "instead of the curated 15-problem benchmark.")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--max-iter-gd", type=int, default=5000)
    parser.add_argument("--max-iter-newton", type=int, default=1000)
    parser.add_argument("--max-iter-bfgs", type=int, default=2000)
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.problems:
        names = args.problems
    elif args.all:
        names = list_problems(args.limit)
    else:
        names = list(BENCHMARK_PROBLEMS)
        if args.limit is not None:
            names = names[: args.limit]

    names = filter_available(names)
    print(f"Benchmarking {len(names)} problems: {names}")

    all_records: list[Record] = []
    all_histories: dict = {}
    for name in names:
        print(f"\n=== {name} ===")
        try:
            records, histories = run_one(
                name,
                max_iter_gd=args.max_iter_gd,
                max_iter_newton=args.max_iter_newton,
                max_iter_bfgs=args.max_iter_bfgs,
            )
            all_records.extend(records)
            all_histories[name] = histories
        except Exception as e:
            print(f"  !! failed to import/run {name}: {e!r}")

    csv_path = os.path.join(args.out_dir, "benchmark.csv")
    with open(csv_path, "w") as fh:
        fh.write("problem,solver,n,iterations,runtime_s,f_final,grad_norm,cond_hessian,converged\n")
        for r in all_records:
            fh.write(
                f"{r.problem},{r.solver},{r.n},{r.iterations},"
                f"{r.runtime_s:.6f},{r.f_final:.10e},{r.grad_norm:.6e},"
                f"{r.cond_hessian:.6e},{r.converged}\n"
            )
    pkl_path = os.path.join(args.out_dir, "histories.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(all_histories, fh)

    print(f"\nWrote {csv_path} and {pkl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
