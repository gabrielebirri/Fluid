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

@dataclass
class SimConfig:
    """Tutti i parametri della simulazione in un unico posto."""

    # Dominio
    resolution_x: int = 224
    resolution_y: int = 224
    domain_x: float = 2.0          # larghezza fisica  [m]
    domain_y: float = 1.5          # altezza fisica    [m]

    # Tempo
    num_steps: int = 500
    dt: float = 0.02               # passo temporale   [s]

    # Fisica
    buoyancy_factor: float = 0.1   # forza di galleggiamento del fumo
    viscosity: float = 0.0001       # viscosità cinematica
    inflow_radius: float = 0.08    # raggio della sorgente di fumo
    inflow_velocity_x: float = 5.0  # velocità iniziale getto di fluido [m/s]

    # Configurazione Random
    seed: int = 42

    # Output
    output_dir: str = "sim_output"
    colormap: str = "plasma"

    # Solutore pressione
    solver_tol: float = 1e-3       # tolleranza CG per make_incompressible
    solver_max_iter: int = 1000    # iterazioni massime

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


def generate_random_config() -> SimConfig:
    """
    Genera una configurazione con parametri fisici casuali.
    Estrapola i valori (e il seed downstream) utilizzando esplicitamente
    il generatore di numeri casuali di PyTorch, permettendo la riproducibilità.
    """
    # 1. Estrae un seed univoco da PyTorch per derivarne gli stati degli altri framework
    new_seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    
    # 2. Genera una velocità di lancio casuale compresa ad es. tra 2.0 e 8.0 usando PyTorch
    vel_x = float(torch.empty(1).uniform_(2.0, 8.0).item())
    
    return SimConfig(
        seed=new_seed,
        inflow_velocity_x=vel_x,
    )