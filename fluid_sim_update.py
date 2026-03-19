"""
Simulazione fluidodinamica modulare con PhiFlow
================================================
Struttura modulare:
  - config.py-style  →  SimConfig (dataclass)
  - domain           →  build_domain()
  - initial cond.    →  init_velocity(), init_smoke()
  - physics step     →  simulation_step()
  - output           →  save_frame(), plot_frame(), save_summary()
  - main loop        →  run_simulation()
"""

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


# Device configuration
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


# ─────────────────────────────────────────────
# 1.  CONFIGURAZIONE
# ─────────────────────────────────────────────

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

    # Configurazione Random
    seed: int = field(default_factory=lambda: int(torch.initial_seed() % (2**32)))

    # Output
    output_dir: str = "sim_output"
    colormap: str = "plasma"

    # Solutore pressione
    solver_tol: float = 1e-3       # tolleranza CG per make_incompressible
    solver_max_iter: int = 1000    # iterazioni massime

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


# ─────────────────────────────────────────────
# 2.  DOMINIO  &  OSTACOLI
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# 3.  CONDIZIONI INIZIALI
# ─────────────────────────────────────────────

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
    return velocity_mask * (5.0, 0.0)


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


# ─────────────────────────────────────────────
# 4.  PASSO DI SIMULAZIONE
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# 5.  OUTPUT
# ─────────────────────────────────────────────

def plot_frame(
    smoke: CenteredGrid,
    velocity: StaggeredGrid,
    step: int,
    cfg: SimConfig,
) -> None:
    """Salva un'immagine del frame corrente (fumo + streamlines)."""

    # Estrai array NumPy
    smoke_np = smoke.values.numpy("y,x")               # shape (Ny, Nx)
    # Ricampiona la velocità sulla griglia centrata per avere forma (Ny, Nx, 2)
    vel_centered = velocity.at(smoke)
    vel_arr = vel_centered.values.numpy("y,x,vector")
    vx = vel_arr[..., 0]
    vy = vel_arr[..., 1]

    # Griglia per le streamlines
    ny, nx = smoke_np.shape
    xs = np.linspace(0, cfg.domain_x, nx)
    ys = np.linspace(0, cfg.domain_y, ny)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)

    # Campo del fumo
    im = ax.imshow(
        smoke_np,
        origin="lower",
        extent=[0, cfg.domain_x, 0, cfg.domain_y],
        cmap=cfg.colormap,
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Concentrazione fumo")

    # Frecce di velocità (sottocampionate)
    step_x = max(1, nx // 20)
    step_y = max(1, ny // 15)
    xs_s = xs[::step_x]
    ys_s = ys[::step_y]
    vx_s = vx[::step_y, ::step_x]
    vy_s = vy[::step_y, ::step_x]
    Xs, Ys = np.meshgrid(xs_s, ys_s)
    ax.quiver(
        Xs, Ys, vx_s, vy_s,
        color="white", alpha=0.55,
        scale=12, width=0.003,
    )

    # Ostacoli drawing in plot_frame rimosso temporaneamente per supportare molteplici figure complesse.
    # L'utente sta ispezionando il tensore output.

    ax.set_title(f"Simulazione fluida — step {step:04d}  (t = {step * cfg.dt:.2f} s)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    path = os.path.join(cfg.output_dir, f"frame_{step:04d}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_single_tensor_frame(tensor_path: str, t_idx: int = -1, colormap: str = "plasma") -> None:
    """Carica il tensore numpy salvato ('T, 3, Ny, Nx') e visualizza un frame interattivo a scelta."""
    if not os.path.exists(tensor_path):
        print(f"[Errore] File {tensor_path} non trovato.")
        return
        
    tensor = np.load(tensor_path)
    T, C, Ny, Nx = tensor.shape
    
    # Gestisci indici negativi o fuori range
    if t_idx < 0:
        t_idx += T
    t_idx = max(0, min(t_idx, T - 1))
    
    print(f"Visualizzazione interattiva frame {t_idx+1}/{T} {tensor.shape} (Chiudi la finestra per proseguire)")
    
    u = tensor[t_idx, 0, :, :]
    v = tensor[t_idx, 1, :, :]
    p = tensor[t_idx, 2, :, :]
    
    speed = np.sqrt(u**2 + v**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    im_speed = axes[0].imshow(speed, origin="lower", cmap=colormap, aspect="auto")
    axes[0].set_title(f"Velocità (Magnitudo) - Step {t_idx+1:04d}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im_speed, ax=axes[0])
    
    im_p = axes[1].imshow(p, origin="lower", cmap="viridis", aspect="auto")
    axes[1].set_title(f"Pressione - Step {t_idx+1:04d}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im_p, ax=axes[1])
    
    plt.tight_layout()
    plt.show()


def save_summary(stats: List[dict], cfg: SimConfig) -> None:
    """Salva un grafico riassuntivo delle statistiche temporali."""
    steps   = [s["step"] for s in stats]
    max_v   = [s["max_velocity"] for s in stats]
    avg_smk = [s["avg_smoke"] for s in stats]
    elapsed = [s["elapsed_s"] for s in stats]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(steps, max_v,   color="royalblue",  lw=2)
    axes[0].set_ylabel("Velocità massima [m/s]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, avg_smk, color="darkorange", lw=2)
    axes[1].set_ylabel("Concentrazione media fumo")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, elapsed, color="seagreen",   lw=2)
    axes[2].set_ylabel("Tempo di calcolo [s]")
    axes[2].set_xlabel("Step")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Statistiche simulazione fluidodinamica", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(cfg.output_dir, "statistics.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[summary] Statistiche salvate → {path}")


# ─────────────────────────────────────────────
# 6.  LOOP PRINCIPALE
# ─────────────────────────────────────────────

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

    print("=" * 55)
    print("  SIMULAZIONE FLUIDODINAMICA — PhiFlow")
    print(f"  Risoluzione : {cfg.resolution_x} × {cfg.resolution_y}")
    print(f"  Steps       : {cfg.num_steps}  |  dt = {cfg.dt} s")
    print(f"  Output dir  : {cfg.output_dir}")
    print("=" * 55)

    # Inizializzazione
    bounds    = build_domain_bounds(cfg)
    obstacles = build_obstacles(cfg)
    smoke     = init_smoke(cfg, bounds)
    velocity = init_velocity(cfg, bounds)
    inflow   = _inflow_field(cfg, bounds)

    stats: List[dict] = []
    tensor_frames = []
    total_start = time.perf_counter()

    for step in range(cfg.num_steps):
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

        print(
            f"  step {step+1:04d}/{cfg.num_steps}"
            f"  |  v_max={s['max_velocity']:.4f} m/s"
            f"  |  smoke_avg={s['avg_smoke']:.4f}"
            f"  |  dt_calc={elapsed*1000:.1f} ms"
        )

    total = time.perf_counter() - total_start
    print("-" * 55)
    print(f"  Simulazione completata in {total:.1f} s")

    save_summary(stats, cfg)
    
    # Save the final tensor
    final_tensor = np.stack(tensor_frames, axis=0)
    out_path = os.path.join(cfg.output_dir, tensor_name)
    np.save(out_path, final_tensor)
    
    print(f"  Tensore simulazione salvato in: {out_path} con shape {final_tensor.shape}")
    print("=" * 55)


# ─────────────────────────────────────────────
# 7.  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Personalizza la configurazione qui oppure lascia i valori di default
    config = SimConfig(
        resolution_x=224,
        resolution_y=224,
        num_steps=50,
        dt=0.02,
        buoyancy_factor=0.1,
        viscosity=0.0001,
        seed=int(torch.initial_seed() % (2**32)),
        output_dir="sim_output",
        colormap="plasma",
    )

    torch.manual_seed(42)

    tensor_filename = "simulation_tensor.npy"  # Modificabile a piacere
    run_simulation(config, tensor_name=tensor_filename)

    # Visualizza a schermo l'ultimo timeframe generato dal tensore (puoi cambiare index qui)
    tensor_file = os.path.join(config.output_dir, tensor_filename)
    plot_single_tensor_frame(tensor_file, t_idx=-1, colormap=config.colormap)
