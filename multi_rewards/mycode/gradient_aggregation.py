"""Gradient aggregation utilities (PCGrad/MGDA/CAGrad/Nash-style)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def _dot_grads(grads_a: Sequence[torch.Tensor], grads_b: Sequence[torch.Tensor]) -> torch.Tensor:
    dot = None
    for g_a, g_b in zip(grads_a, grads_b):
        if g_a is None or g_b is None:
            continue
        val = (g_a * g_b).sum()
        dot = val if dot is None else dot + val
    if dot is None:
        dot = torch.tensor(0.0)
    return dot


def _norm_sq(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    return _dot_grads(grads, grads)


def pcgrad(
    grads_list: List[List[torch.Tensor]],
    *,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> List[torch.Tensor]:
    """Project conflicting gradients away from each other (PCGrad)."""
    num_tasks = len(grads_list)
    if num_tasks == 0:
        return []

    order = torch.randperm(num_tasks, generator=generator).tolist() if shuffle else list(range(num_tasks))

    adjusted = []
    for i in range(num_tasks):
        g_i = [g.clone() if g is not None else None for g in grads_list[i]]
        for j in order:
            if i == j:
                continue
            g_j = grads_list[j]
            dot = _dot_grads(g_i, g_j)
            if dot.item() < 0:
                denom = _norm_sq(g_j).clamp(min=1e-12)
                for k, g in enumerate(g_i):
                    if g is None or g_j[k] is None:
                        continue
                    g_i[k] = g - (dot / denom) * g_j[k]
        adjusted.append(g_i)

    # Aggregate by mean
    agg = []
    for params in zip(*adjusted):
        tensors = [g for g in params if g is not None]
        if not tensors:
            agg.append(None)
        else:
            agg.append(torch.stack(tensors, dim=0).mean(dim=0))
    return agg


def _project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Project onto the probability simplex."""
    if v.numel() == 1:
        return torch.ones_like(v)
    v_sorted, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(v_sorted, dim=0) - 1
    idx = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = v_sorted - cssv / idx > 0
    rho = idx[cond][-1]
    theta = cssv[cond][-1] / rho
    w = torch.clamp(v - theta, min=0)
    return w


def mgda_weights(
    grads_list: List[List[torch.Tensor]],
    *,
    max_iters: int = 50,
    lr: float | None = None,
) -> torch.Tensor:
    """Approximate MGDA weights by projected gradient descent."""
    num_tasks = len(grads_list)
    if num_tasks == 0:
        return torch.tensor([])
    if num_tasks == 1:
        return torch.ones(1)

    device = None
    for grads in grads_list:
        for g in grads:
            if g is not None:
                device = g.device
                break
        if device is not None:
            break
    if device is None:
        device = torch.device("cpu")
    G = torch.zeros((num_tasks, num_tasks), device=device)
    for i in range(num_tasks):
        for j in range(num_tasks):
            G[i, j] = _dot_grads(grads_list[i], grads_list[j])

    w = torch.full((num_tasks,), 1.0 / num_tasks, device=device)
    step = lr
    if step is None:
        step = 1.0 / (G.abs().max().clamp(min=1.0))

    for _ in range(max_iters):
        grad = 2 * G.mv(w)
        w = _project_simplex(w - step * grad)
    return w


def cagrad_weights(
    grads_list: List[List[torch.Tensor]],
    *,
    alpha: float = 0.5,
    max_iters: int = 50,
) -> torch.Tensor:
    """Conflict-averse weights (regularized MGDA approximation)."""
    w = mgda_weights(grads_list, max_iters=max_iters)
    if w.numel() == 0:
        return w
    uniform = torch.full_like(w, 1.0 / w.numel())
    blended = (1.0 - alpha) * w + alpha * uniform
    return _project_simplex(blended)


def nash_mtl_weights(
    grads_list: List[List[torch.Tensor]],
    *,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Approximate Nash-MTL weights via inverse Gram matrix heuristic."""
    num_tasks = len(grads_list)
    if num_tasks == 0:
        return torch.tensor([])
    if num_tasks == 1:
        return torch.ones(1)

    device = None
    for grads in grads_list:
        for g in grads:
            if g is not None:
                device = g.device
                break
        if device is not None:
            break
    if device is None:
        device = torch.device("cpu")
    G = torch.zeros((num_tasks, num_tasks), device=device)
    for i in range(num_tasks):
        for j in range(num_tasks):
            G[i, j] = _dot_grads(grads_list[i], grads_list[j])

    G = G + eps * torch.eye(num_tasks, device=device)
    ones = torch.ones(num_tasks, device=device)
    try:
        w = torch.linalg.solve(G, ones)
    except Exception:
        w = torch.ones(num_tasks, device=device)
    w = _project_simplex(w)
    return w


def aggregate_gradients(
    grads_list: List[List[torch.Tensor]],
    method: str,
    *,
    pcgrad_shuffle: bool = False,
    cagrad_alpha: float = 0.5,
    mgda_iters: int = 50,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Return aggregated gradients and the weight vector (if applicable)."""
    method = method.lower()
    weights = None
    if method == "pcgrad":
        agg = pcgrad(grads_list, shuffle=pcgrad_shuffle)
        weights = torch.tensor([])
        return agg, weights
    if method == "mgda":
        weights = mgda_weights(grads_list, max_iters=mgda_iters)
    elif method == "cagrad":
        weights = cagrad_weights(grads_list, alpha=cagrad_alpha, max_iters=mgda_iters)
    elif method in {"nash", "nash_mtl"}:
        weights = nash_mtl_weights(grads_list)
    else:
        raise ValueError(f"Unknown gradient aggregation method: {method}")

    agg = []
    for params in zip(*grads_list):
        tensors = [g for g in params if g is not None]
        if not tensors:
            agg.append(None)
        else:
            stacked = torch.stack(tensors, dim=0)
            w = weights.to(stacked.device).view(-1, 1)
            while w.dim() < stacked.dim():
                w = w.unsqueeze(-1)
            agg.append((stacked * w).sum(dim=0))
    return agg, weights
