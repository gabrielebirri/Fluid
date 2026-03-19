from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math

import torch
import numpy as np
import matplotlib
from tqdm.auto import tqdm
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from phi.torch.flow import (
    CenteredGrid, StaggeredGrid, Box, Sphere, Obstacle,
    extrapolation, fluid, advect, diffuse,
    spatial, batch,
)
import phi.math as math

from src.config import SimConfig
from src.domain import build_domain_bounds, build_obstacles
from src.initial_conditions import init_smoke, init_velocity, _inflow_field
from src.solver import simulation_step
from src.visuals import save_summary


def collect_stats(
    step: int,
    smoke: CenteredGrid,
    velocity: StaggeredGrid,
    elapsed: float,
) -> dict:
    """Raccoglie metriche scalari per il frame corrente."""
    vel_arr = velocity.staggered_tensor().numpy("y,x,vector")
    speed = np.sqrt(vel_arr[..., 0] ** 2 + vel_arr[..., 1] ** 2)
    smoke_np = smoke.values.numpy("y,x")
    return {
        "step":         step,
        "max_velocity": float(speed.max()),
        "avg_smoke":    float(smoke_np.mean()),
        "elapsed_s":    elapsed,
    }


def run_simulation(cfg: Optional[SimConfig] = None, tensor_name: str = "simulation_tensor.npy", num_steps: Optional[int] = None) -> None:
    """Entry-point principale della simulazione."""
    if cfg is None:
        cfg = SimConfig()

    if num_steps is not None:
        cfg.num_steps = num_steps

    print("=" * 75)
    print(f"  Res: {cfg.resolution_x}x{cfg.resolution_y} | Steps: {cfg.num_steps} | dt: {cfg.dt}s | Visc: {cfg.viscosity:.6f}")
    print(f"  Out: {cfg.output_dir}")
    print("=" * 75)

    # Inizializzazione
    bounds    = build_domain_bounds(cfg)
    obstacles = build_obstacles(cfg)
    smoke     = init_smoke(cfg, bounds)
    velocity = init_velocity(cfg, bounds)
    inflow   = _inflow_field(cfg, bounds)

    stats: List[dict] = []
    tensor_frames = []
    total_start = time.perf_counter()

    pbar = tqdm(range(cfg.num_steps), desc="Fluid Sim Progress")
    for step in pbar:
        t0 = time.perf_counter()

        smoke, velocity, pressure = simulation_step(smoke, velocity, inflow, obstacles, cfg)

        # Extract u, v, p fields on centered grid and stack to tensor
        vel_centered = velocity.at(smoke)
        vel_arr = vel_centered.values.numpy("y,x,vector")
        u = vel_arr[..., 0]
        v = vel_arr[..., 1]
        p_arr = pressure.values.numpy("y,x")
        
        frame_tensor = np.stack([u, v, p_arr], axis=0)
        tensor_frames.append(frame_tensor)

        elapsed = time.perf_counter() - t0
        s = collect_stats(step + 1, smoke, velocity, elapsed)
        stats.append(s)

        pbar.set_postfix({
            'v_max': f"{s['max_velocity']:.4f}",
            'smoke': f"{s['avg_smoke']:.4f}",
            'dt_ms': f"{elapsed*1000:.1f}"
        })

    total = time.perf_counter() - total_start
    
    # Save the final tensor
    final_tensor = np.stack(tensor_frames, axis=0)
    out_path = os.path.join(cfg.output_dir, tensor_name)
    np.save(out_path, final_tensor)
    
    print(f"  Tensor saved in: {out_path} with shape {final_tensor.shape}")

