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


# --------------------------------------------------------------------------------------------------
# Consensus ADMM across regions (spec §11; lecture: ADMM for distributed decision making)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ADMMSolution:
    manifesto: torch.Tensor          # consensus z ∈ Δ  [5]
    share: float                     # exact expected share of z over all voters
    local: torch.Tensor              # final local copies x_r  [R, 5]
    primal_residuals: list[float]    # ‖x − 1⊗z‖ per iteration
    dual_residuals: list[float]      # ρ √R ‖z − z_prev‖ per iteration
    iterations: int


def admm_manifesto(v: torch.Tensor, region_of: torch.Tensor, sensitivity: float, threshold: float, *,
                   rho: float = 5.0, iterations: int = 300, inner_steps: int = 10, inner_step: float = 0.05,
                   smoothing: float = 0.02, start: torch.Tensor | None = None, tol: float = 1e-6,
                   num_regions: int = 4) -> ADMMSolution:
    """Consensus ADMM:  min Σ_r f_r(x_r) + I_Δ(z)  s.t.  x_r = z,  with f_r(x) = −(n_r/N)·F_r(x).

    F_r is region r's (smoothed) expected share, so Σ_r f_r = −(overall expected share) and the
    consensus solution coincides with the centralised projected-gradient problem.

    Scaled-form updates (u = λ/ρ):
      x_r ← argmin_x f_r(x) + ρ/2 ‖x − z + u_r‖²      (local; inexact: ``inner_steps`` gradient steps)
      z   ← P_Δ( mean_r (x_r + u_r) )                  (prox of I_Δ = projection; the consensus step)
      u_r ← u_r + x_r − z                               (dual ascent: accumulated disagreement)
    The local objective is non-convex, so ADMM reaches a stationary point; callers may multi-start.
    Non-convex ADMM needs ρ above the curvature of f_r: with the sharp surrogate (δ = 1e-3) the iterates
    oscillate, so the default uses δ = 0.02 and ρ = 5 (converges in ≈ 40 iterations on 3k voters,
    within ≈ 1e-3 expected share of the centralised optimum; the gap is the smoothing bias).
    """
    v = v.detach()
    dtype, device = v.dtype, v.device
    region_of = region_of.to(device)
    weights = torch.stack([(region_of == r).sum() for r in range(num_regions)]).to(dtype) / v.shape[0]
    groups = [v[region_of == r] for r in range(num_regions)]
    z = (torch.full((v.shape[-1],), 1.0 / v.shape[-1], dtype=dtype, device=device)
         if start is None else project_simplex(start.to(v)))
    x = z.expand(num_regions, -1).clone()
    u = torch.zeros_like(x)
    primal, dual = [], []
    used = 0
    for used in range(1, iterations + 1):
        anchor = z - u                                   # x_r is pulled towards z − u_r
        for r, group in enumerate(groups):
            if group.shape[0] == 0:
                x[r] = anchor[r]
                continue
            xr = x[r].clone()
            for _ in range(inner_steps):
                xr.requires_grad_(True)
                loss = -weights[r] * expected_share(group, xr, sensitivity, threshold, smoothing) \
                    + 0.5 * rho * torch.sum((xr - anchor[r]) ** 2)
                (grad,) = torch.autograd.grad(loss, xr)
                xr = (xr - inner_step * grad).detach()
            x[r] = xr
        z_prev = z
        z = project_simplex((x + u).mean(0))
        u = u + x - z
        primal.append(float(torch.linalg.vector_norm(x - z)))
        dual.append(float(rho * num_regions ** 0.5 * torch.linalg.vector_norm(z - z_prev)))
        if primal[-1] < tol and dual[-1] < tol:
            break
    share = float(expected_share(v, z, sensitivity, threshold, 0.0))
    return ADMMSolution(z, share, x, primal, dual, used)
