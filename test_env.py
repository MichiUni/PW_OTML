"""Sanity check for the pycutest + scipy environment.

Filters CUTEst for smooth unconstrained problems with 'other' objective type,
loads a sample instance, and solves it with scipy L-BFGS-B to confirm that
objective, gradient, and Hessian evaluations all work end-to-end.
"""

from __future__ import annotations

import sys

import numpy as np
import pycutest
from scipy.optimize import minimize


def pick_problem(candidates: list[str], filtered: list[str]) -> str:
    """Return the first candidate present in the filter list, else filtered[0]."""
    for name in candidates:
        if name in filtered:
            return name
    return filtered[0]


def main() -> int:
    filtered = pycutest.find_problems(
        objective="other",
        constraints="unconstrained",
        regular=True,
    )
    filtered = sorted(filtered)
    print(f"pycutest.find_problems -> {len(filtered)} matches")
    print(f"first 10: {filtered[:10]}")

    # Prefer small, parameter-free instances so the sanity check stays fast.
    preferred = ["HIMMELBG", "HIMMELBH", "HAIRY", "BEALE", "GULF", "SNAIL"]
    problem_name = pick_problem(preferred, filtered)
    print(f"\nSelected problem: {problem_name}")

    problem = pycutest.import_problem(problem_name)
    x0 = problem.x0.copy()
    n = problem.n

    # Verify objective, gradient, and Hessian all evaluate at x0.
    f0, g0 = problem.obj(x0, gradient=True)
    H0 = problem.hess(x0)
    print(f"dim n           : {n}")
    print(f"f(x0)           : {f0:.6e}")
    print(f"||grad f(x0)||  : {np.linalg.norm(g0):.6e}")
    print(f"Hessian shape   : {H0.shape}")

    result = minimize(
        fun=lambda x: problem.obj(x, gradient=True),
        x0=x0,
        jac=True,
        method="L-BFGS-B",
        options={"gtol": 1e-8, "ftol": 1e-12, "maxiter": 1000},
    )

    print("\n--- L-BFGS-B result ---")
    print(f"problem         : {problem_name}")
    print(f"dim n           : {n}")
    print(f"f(x*)           : {result.fun:.10e}")
    print(f"||grad f(x*)||  : {np.linalg.norm(result.jac):.6e}")
    print(f"iterations      : {result.nit}")
    print(f"func evals      : {result.nfev}")
    print(f"success flag    : {result.success}")
    print(f"status message  : {result.message}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())