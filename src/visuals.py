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

