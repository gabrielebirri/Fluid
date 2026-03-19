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


def init_smoke(cfg: SimConfig, bounds: Box) -> CenteredGrid:
    """Griglia scalare del fumo (concentrazione ∈ [0,1])."""
    smoke = CenteredGrid(
        0.0,
        extrapolation.BOUNDARY,
        x=cfg.resolution_x,
        y=cfg.resolution_y,
        bounds=bounds,
    )
    return smoke


def init_velocity(cfg: SimConfig, bounds: Box) -> StaggeredGrid:
    """Campo vettoriale di velocità iniziale (flusso da sinistra verso destra)."""
    # Imposta una velocità iniziale uniforme su un'ampia porzione del margine sinistro
    left_margin = Box(x=(0, cfg.domain_x * 0.20), y=(0, cfg.domain_y))
    velocity_mask = StaggeredGrid(
        left_margin,
        extrapolation.combine_sides(x=extrapolation.BOUNDARY, y=extrapolation.ZERO),  # x aperto (flusso continuo), y chiuso
        x=cfg.resolution_x,
        y=cfg.resolution_y,
        bounds=bounds,
    )
    return velocity_mask * (cfg.inflow_velocity_x, 0.0)


def _inflow_field(cfg: SimConfig, bounds: Box) -> CenteredGrid:
    """Sorgente di fumo estesa lungo tutto l'asse y a sinistra."""
    # Sostituiamo la piccola Sfera (lineare) con una Box che copre (quasi) tutto l'asse y
    inflow_box = Box(x=(0.0, 0.15), y=(0.1, cfg.domain_y - 0.1))
    inflow = CenteredGrid(
        inflow_box,
        extrapolation.BOUNDARY,
        x=cfg.resolution_x,
        y=cfg.resolution_y,
        bounds=bounds,
    )
    return inflow

