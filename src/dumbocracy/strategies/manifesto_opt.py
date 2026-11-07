"""Manifesto optimisation on the simplex (spec §11; lectures: proximal operator, projected gradient).

Maximise the expected vote share  f(b) = mean_i σ(β (1 − ½‖v_i − b‖₁ − z))  over b ∈ Δ⁴.
The TV distance is non-smooth, so |x| is replaced by the pseudo-Huber surrogate √(x² + δ²) − δ;
each iteration is a gradient step followed by the Euclidean projection onto the simplex, which is
exactly the proximal operator of the simplex indicator (prox_{I_Δ} = P_Δ). The objective is not
concave, so several starts are used and the best is chosen on the *exact* (non-smoothed) objective.

P4 extends this module with posterior-sample inputs and the consensus-ADMM split across regions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def project_simplex(y: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of each row of ``y`` (last dim) onto the probability simplex.

    Sort-based algorithm (Held et al. 1974; Duchi et al. 2008): θ = (Σ_{k≤ρ} u_k − 1)/ρ with
    ρ = max{k : u_k − (Σ_{j≤k} u_j − 1)/k > 0}, x = max(y − θ, 0).
    """
    u, _ = torch.sort(y, dim=-1, descending=True)
    css = torch.cumsum(u, dim=-1) - 1.0
    k = torch.arange(1, y.shape[-1] + 1, device=y.device, dtype=y.dtype)
    condition = u - css / k > 0
    rho = condition.to(y.dtype).cumsum(-1).argmax(-1, keepdim=True)  # last index where condition holds
    theta = css.gather(-1, rho) / (rho + 1).to(y.dtype)
    return torch.clamp(y - theta, min=0.0)


def expected_share(v: torch.Tensor, b: torch.Tensor, sensitivity: float, threshold: float,
                   smoothing: float = 0.0, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Expected vote share of manifesto(s) ``b`` ([5] or [M, 5]) over voters ``v`` [N, 5].

    ``smoothing`` > 0 uses the pseudo-Huber surrogate (for gradients); 0 gives the exact share.
    ``weights`` [N] optionally re-weights voters (e.g. posterior samples / region masses).
    """
    diff = v.unsqueeze(-3) - b.unsqueeze(-2) if b.dim() == 2 else v - b
    absdiff = torch.sqrt(diff * diff + smoothing ** 2) - smoothing if smoothing > 0 else diff.abs()
    similarity = 1.0 - 0.5 * absdiff.sum(-1)
    prob = torch.sigmoid(sensitivity * (similarity - threshold))
    if weights is None:
        return prob.mean(-1)
    return (prob * weights).sum(-1) / weights.sum()


@dataclass(frozen=True)
class ManifestoSolution:
    manifesto: torch.Tensor   # [5] on the simplex
    share: float              # exact expected share at the solution
    kkt_residual: float       # ‖b − P_Δ(b + η∇f(b))‖ / η at the smoothed objective
    iterations: int


def default_starts(v: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
    """Uniform, population mean, the 5 vertices pulled to the centre, and optional extra starts."""
    k = v.shape[-1]
    uniform = torch.full((1, k), 1.0 / k, dtype=v.dtype, device=v.device)
    mean = v.mean(0, keepdim=True)
    vertices = 0.6 * torch.eye(k, dtype=v.dtype, device=v.device) + 0.4 / k
    starts = [uniform, mean, vertices]
    if extra is not None:
        starts.append(extra.to(v))
    return project_simplex(torch.cat(starts))


def optimise_manifesto(v: torch.Tensor, sensitivity: float, threshold: float, *,
                       starts: torch.Tensor | None = None, weights: torch.Tensor | None = None,
                       step: float = 0.02, iterations: int = 400, smoothing: float = 1e-3,
                       tol: float = 1e-7) -> ManifestoSolution:
    """Multi-start projected gradient ascent on the smoothed expected share (all starts batched)."""
    v = v.detach()
    b = (default_starts(v) if starts is None else project_simplex(starts.to(v))).clone()
    used = 0
    for used in range(1, iterations + 1):
        b.requires_grad_(True)
        objective = expected_share(v, b, sensitivity, threshold, smoothing, weights).sum()
        (grad,) = torch.autograd.grad(objective, b)
        with torch.no_grad():
            new_b = project_simplex(b + step * grad)
            moved = float((new_b - b).abs().max())
            b = new_b
        if moved < tol:
            break
    with torch.no_grad():
        exact = expected_share(v, b, sensitivity, threshold, 0.0, weights)
        best = int(torch.argmax(exact))
        best_b = b[best].detach()
    best_b.requires_grad_(True)
    objective = expected_share(v, best_b, sensitivity, threshold, smoothing, weights)
    (grad,) = torch.autograd.grad(objective, best_b)
    with torch.no_grad():
        residual = float(torch.linalg.vector_norm(best_b - project_simplex(best_b + step * grad)) / step)
    return ManifestoSolution(best_b.detach(), float(exact[best]), residual, used)
