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

from phi.flow import (
    CenteredGrid, StaggeredGrid, Box, Sphere, Obstacle,
    extrapolation, fluid, advect, diffuse,
    spatial, batch,
)
import phi.math as math

from src.config import SimConfig


def build_obstacles(cfg: SimConfig) -> tuple:
    """Crea ostacoli casuali (Box o Sphere), anche più forme, con posizione e orientazione casuale."""
    import random
    
    current_seed = cfg.seed if cfg.seed is not None else int(torch.initial_seed() % (2**32))
    np.random.seed(current_seed)
    random.seed(current_seed)
    
    num_obstacles = random.randint(2, 4)
    obstacles = []
    
    for i in range(num_obstacles):
        shape_type = random.choice(["sphere", "box"])
        
        # Scegli centro
        cx = random.uniform(cfg.domain_x * 0.2, cfg.domain_x * 0.8)
        cy = random.uniform(cfg.domain_y * 0.2, cfg.domain_y * 0.8)
        
        if shape_type == "sphere":
            radius = random.uniform(0.05, 0.2)
            geom = Sphere(x=cx, y=cy, radius=radius)
            print(f"  Ostacolo {i+1} : Sfera (centro={cx:.2f}, {cy:.2f} | r={radius:.2f})")
        else:
            width = random.uniform(0.1, 0.4)
            height = random.uniform(0.1, 0.4)
            angle = random.uniform(0, math.pi)
            geom = Box(x=(cx - width/2, cx + width/2), y=(cy - height/2, cy + height/2)).rotated(angle)
            print(f"  Ostacolo {i+1} : Box (centro={cx:.2f}, {cy:.2f} | {width:.2f}×{height:.2f} | rot={angle:.2f})")
            
        obstacles.append(Obstacle(geom))
        
    return tuple(obstacles)


def build_domain_bounds(cfg: SimConfig) -> Box:
    """Restituisce la box del dominio fisico."""
    return Box(x=(0, cfg.domain_x), y=(0, cfg.domain_y))

