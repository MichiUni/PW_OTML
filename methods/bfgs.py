"""BFGS Quasi-Newton method with SciPy's Wolfe line search."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.optimize import line_search


def bfgs(
    obj_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    x0: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 2000,
    c1: float = 1e-4,
    c2: float = 0.9,
) -> dict:
    """Minimize with BFGS using SciPy's Wolfe line search.

    The inverse-Hessian approximation `H_inv` is updated with the standard
    BFGS formula:  H_{k+1} = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T,
    with rho = 1 / (y^T s). Steps where y^T s <= 0 skip the update to keep
    H_inv positive definite (the Wolfe conditions normally prevent this,
    but line-search failure can still occur).
    """
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    I = np.eye(n)

    f_only = lambda z: obj_grad(z)[0]
    g_only = lambda z: obj_grad(z)[1]

    fx, grad = obj_grad(x)
    H_inv = I.copy()

    f_history = [float(fx)]
    grad_history = [float(np.linalg.norm(grad))]
    ls_failures = 0

    converged = False
    k = 0
    for k in range(1, max_iter + 1):
        gnorm = np.linalg.norm(grad)
        if gnorm <= tol:
            converged = True
            break

        direction = -H_inv @ grad
        # Guard against a bad direction (e.g. numerical loss of PD).
        if not np.all(np.isfinite(direction)) or float(np.dot(grad, direction)) >= 0.0:
            H_inv = I.copy()
            direction = -grad

        ls = line_search(
            f_only, g_only, x, direction,
            gfk=grad, old_fval=fx,
            c1=c1, c2=c2, maxiter=50,
        )
        alpha = ls[0]

        if alpha is None or alpha == 0.0 or not np.isfinite(alpha):
            # Reset the approximation and retry with steepest descent once.
            ls_failures += 1
            H_inv = I.copy()
            direction = -grad
            ls = line_search(
                f_only, g_only, x, direction,
                gfk=grad, old_fval=fx,
                c1=c1, c2=c2, maxiter=50,
            )
            alpha = ls[0]
            if alpha is None or alpha == 0.0 or not np.isfinite(alpha):
                break

        s = alpha * direction
        x_new = x + s

        fx_new, grad_new = obj_grad(x_new)
        y = grad_new - grad

        ys = float(np.dot(y, s))
        if ys > 1e-12:
            rho = 1.0 / ys
            V = I - rho * np.outer(s, y)
            H_inv = V @ H_inv @ V.T + rho * np.outer(s, s)
        # If curvature condition violates PD, skip the update.

        x, fx, grad = x_new, fx_new, grad_new
        f_history.append(float(fx))
        grad_history.append(float(np.linalg.norm(grad)))

    if np.linalg.norm(grad) <= tol:
        converged = True

    return {
        "x": x,
        "f": float(fx),
        "grad_norm": float(np.linalg.norm(grad)),
        "iterations": k,
        "converged": bool(converged),
        "f_history": f_history,
        "grad_history": grad_history,
        "ls_failures": ls_failures,
        "method": "BFGS",
    }
