from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math

import torch
import numpy as np
import matplotlib
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


def simulation_step(
    smoke: CenteredGrid,
    velocity: StaggeredGrid,
    inflow: CenteredGrid,
    obstacles: tuple,
    cfg: SimConfig,
) -> Tuple[CenteredGrid, StaggeredGrid, CenteredGrid]:
    """
    Avanza la simulazione di un passo temporale.

    Pipeline:
        1. Aggiunge fumo dalla sorgente
        2. Applica forza di galleggiamento al campo di velocità
        3. Advecta velocità e fumo
        4. Diffonde la velocità (viscosità)
        5. Rende il campo di velocità privo di divergenza (incomprimibilità)
    """
    # --- sorgente di fumo ---
    smoke = smoke + inflow * cfg.dt

    # --- forza propulsiva (il fumo si sposta verso destra) ---
    # Il fumo è su CenteredGrid → campionarlo sulla StaggeredGrid prima di sommare
    buoyancy_force = (smoke * (cfg.buoyancy_factor, 0.0)).at(velocity)
    velocity = velocity + buoyancy_force * cfg.dt         # integrazione Eulero

    # --- advection ---
    smoke = advect.semi_lagrangian(smoke, velocity, cfg.dt)
    velocity = advect.semi_lagrangian(velocity, velocity, cfg.dt)

    # --- diffusione (viscosità) ---
    velocity = diffuse.explicit(velocity, cfg.viscosity, cfg.dt)

    # --- proiezione: rimuove la divergenza → incomprimibilità ---
    # Usa scipy-direct (robusto) con fallback su CG
    try:
        solve = math.Solve('scipy-direct', rel_tol=cfg.solver_tol, abs_tol=cfg.solver_tol,
                           max_iterations=cfg.solver_max_iter)
        velocity, pressure = fluid.make_incompressible(velocity, obstacles, solve=solve)
    except Exception:
        solve = math.Solve('CG', rel_tol=cfg.solver_tol * 10, abs_tol=cfg.solver_tol * 10,
                           max_iterations=cfg.solver_max_iter * 2)
        velocity, pressure = fluid.make_incompressible(velocity, obstacles, solve=solve)

    # Clamp della velocità per stabilità numerica
    velocity = velocity.with_values(math.clip(velocity.values, -10.0, 10.0))

    return smoke, velocity, pressure
