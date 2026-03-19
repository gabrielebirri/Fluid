import phiflow as pf
import numpy as np
from PIL import Image
import os

def generate_cfd_data_for_vit(
    num_simulations=10,
    timesteps_per_sim=100,
    domain_size=(1.0, 1.0),
    resolution=(64, 64),
    output_dir='cfd_vit_data',
    normalize_velocity=True
):
    """
    Generate CFD simulation data suitable for training a Vision Transformer (ViT) model.
    Each simulation consists of a sequence of images representing fluid flow at different timestamps.

    Args:
        num_simulations (int): Number of independent simulations to generate. Default: 10.
        timesteps_per_sim (int): Number of timestamps per simulation. Default: 100.
        domain_size (tuple): Size of the computational domain (x, y) in meters. Default: (1.0, 1.0).
        resolution (tuple): Spatial resolution of the simulation grid (nx, ny). Default: (64, 64).
        output_dir (str): Directory to save the generated image sequences. Will be created if it doesn't exist.
        normalize_velocity (bool): Whether to normalize velocity values to [0, 1] range for visualization.
                                   If False, uses raw values which may require scaling during training.

    Returns:
        None: Saves simulations as image sequences in the specified directory.
              Each simulation is saved in a subdirectory with PNG images for each timestamp.
              The images are RGB where R=u (x-velocity), G=v (y-velocity), B=magnitude or 0.
    """
    os.makedirs(output_dir, exist_ok=True)

    for sim_idx in range(num_simulations):
        # Set up the problem
        x = pf.GeomRect(*domain_size)
        u = pf.Velocity(x, has_pressure=True)

        # Initialize with random noise (different seed for each simulation)
        u.init_random(seed=sim_idx + 42)

        # Parameters - adjust these based on desired fluid behavior
        params = pf.NavierStokesParameters()
        params.viscosity = 0.1  # Adjust viscosity as needed

        # Solver setup - using Adam solver for time marching
        solver = pf.NavierStokesSolver(u, params)
        solver.time_marching.scheme = 'adam'
        solver.time_marching.adam.beta1 = 0.9
        solver.time_marching.adam.beta2 = 0.999

        # Create directory for this simulation
        sim_dir = os.path.join(output_dir, f'simulation_{sim_idx:03d}')
        os.makedirs(sim_dir, exist_ok=True)

        print(f"Generating simulation {sim_idx + 1}/{num_simulations}...")

        for t in range(timesteps_per_sim):
            # Step the solver forward in time
            solver.step()

            # Get velocity field (shape: [2, nx, ny] - channels first)
            vel_field = u.value.numpy()  # This is a numpy array of shape (2, nx, ny)

            # Process velocity components for visualization
            u_comp = vel_field[0]
            v_comp = vel_field[1]

            if normalize_velocity:
                # Normalize to [0, 1] based on current simulation's min/max
                u_norm = (u_comp - np.min(u_comp)) / (np.max(u_comp) - np.min(u_comp))
                v_norm = (v_comp - np.min(v_comp)) / (np.max(v_comp) - np.min(v_comp))

                # Calculate magnitude for B channel
                magnitude = np.sqrt(u_comp**2 + v_comp**2)
                mag_norm = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
            else:
                u_norm, v_norm = u_comp, v_comp
                # For raw values, you might want to clip or scale differently
                mag_norm = np.sqrt(u_comp**2 + v_comp**2)

            # Create RGB image: R=u, G=v, B=magnitude (or 0 if not using magnitude)
            rgb_image = np.stack([u_norm, v_norm, mag_norm], axis=-1)

            # Convert to uint
