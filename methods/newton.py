"""Newton's method with gradient-related safeguard and Armijo line search."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from .gradient_descent import armijo_backtracking


def _try_newton_direction(
    H: np.ndarray, grad: np.ndarray, c_angle: float
) -> Tuple[np.ndarray, bool]:
    """Attempt to solve H d = -grad and check the gradient-related conditions.

    Returns (direction, ok). If `ok` is False, the caller should fall back to
    a steepest-descent step.
    """
    n = grad.size
    if not np.all(np.isfinite(H)):
        return np.zeros(n), False

    # Prefer a Cholesky solve (confirms SPD); fall back to a generic solve.
    d = None
    try:
        L = np.linalg.cholesky(H)
        y = np.linalg.solve(L, -grad)
        d = np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        try:
            d = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            return np.zeros(n), False

    if not np.all(np.isfinite(d)):
        return d, False

    gTd = float(np.dot(grad, d))
    d_norm = float(np.linalg.norm(d))
    g_norm = float(np.linalg.norm(grad))
    if d_norm == 0.0 or g_norm == 0.0:
        return d, False
    # Descent direction and angle condition: -g^T d >= c * ||g|| * ||d||.
    if gTd >= 0.0:
        return d, False
    if -gTd < c_angle * g_norm * d_norm:
        return d, False
    return d, True


def newton(
    obj_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    hess: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000,
    alpha_init: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    c_angle: float = 1e-4,
) -> dict:
    """Minimize with Newton's method + gradient-related safeguard.

    If the Newton system fails or `d_k` is not gradient-related, this
    iteration falls back to a steepest-descent step. All accepted steps use
    Armijo backtracking.
    """
    x = np.asarray(x0, dtype=float).copy()
    f_only = lambda z: obj_grad(z)[0]

    fx, grad = obj_grad(x)
    f_history = [float(fx)]
    grad_history = [float(np.linalg.norm(grad))]
    fallbacks = 0

    converged = False
    k = 0
    for k in range(1, max_iter + 1):
        gnorm = np.linalg.norm(grad)
        if gnorm <= tol:
            converged = True
            break

        H = hess(x)
        direction, ok = _try_newton_direction(H, grad, c_angle)
        if not ok:
            direction = -grad
            fallbacks += 1

        alpha, _ = armijo_backtracking(
            f_only, x, fx, grad, direction,
            alpha_init=alpha_init, c1=c1, rho=rho,
        )
        if alpha == 0.0:
            # Try a pure GD step as a last resort.
            if ok:
                direction = -grad
                fallbacks += 1
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
        "fallbacks": fallbacks,
        "method": "Newton",
    }
