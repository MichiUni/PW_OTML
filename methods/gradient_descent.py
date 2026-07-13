"""Gradient Descent with Armijo backtracking line search."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def armijo_backtracking(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    fx: float,
    grad: np.ndarray,
    direction: np.ndarray,
    alpha_init: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    max_ls_iter: int = 50,
) -> Tuple[float, int]:
    """Standard Armijo backtracking line search.

    Returns (alpha, n_evals). Returns alpha = 0.0 if the search fails.
    """
    alpha = alpha_init
    slope = float(np.dot(grad, direction))
    n_evals = 0
    for _ in range(max_ls_iter):
        f_trial = f(x + alpha * direction)
        n_evals += 1
        if np.isfinite(f_trial) and f_trial <= fx + c1 * alpha * slope:
            return alpha, n_evals
        alpha *= rho
    return 0.0, n_evals


def gradient_descent(
    obj_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    x0: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 5000,
    alpha_init: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
) -> dict:
    """Minimize using Gradient Descent with Armijo backtracking.

    Stopping rule: ||grad f(x_k)|| <= tol (default 1e-4).
    `obj_grad(x)` must return (f, grad).
    """
    x = np.asarray(x0, dtype=float).copy()
    f_only = lambda z: obj_grad(z)[0]

    fx, grad = obj_grad(x)
    f_history = [float(fx)]
    grad_history = [float(np.linalg.norm(grad))]

    converged = False
    k = 0
    for k in range(1, max_iter + 1):
        gnorm = np.linalg.norm(grad)
        if gnorm <= tol:
            converged = True
            break

        direction = -grad
        alpha, _ = armijo_backtracking(
            f_only, x, fx, grad, direction,
            alpha_init=alpha_init, c1=c1, rho=rho,
        )
        if alpha == 0.0:
            break

        x = x + alpha * direction
        fx, grad = obj_grad(x)
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
        "method": "GD",
    }
