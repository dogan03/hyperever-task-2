# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
CartPole RL environment with custom physical parameters.

Usage:
    ./isaaclab.sh -p scripts/tutorials/03_envs/main.py --num_envs 32
"""

import argparse

import torch
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(
    description="CartPole RL environment with custom parameters"
)
parser.add_argument(
    "--num_envs", type=int, default=16, help="Number of environments to spawn"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching
from cartpole_sim import CartPoleSim


def main():
    """Main function."""

    # Custom physical parameters
    CART_MASS = 1.0
    POLE_MASS = 0.2
    POLE_LENGTH = 0.5
    GRAVITY = 9.81

    # Create simulation
    sim = CartPoleSim(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        cart_mass=CART_MASS,
        pole_mass=POLE_MASS,
        pole_length=POLE_LENGTH,
        gravity=GRAVITY,
    )

    # Simulation loop
    sim.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            # Reset every 300 steps
            if sim.step_count % 300 == 0 and sim.step_count > 0:
                obs = sim.reset()

            # Generate random actions
            actions = sim.get_random_actions()

            # Step environment
            obs, rew, terminated, truncated, info = sim.step(actions)
            print(rew)

            # Print state of first environment (pass obs)
            sim.print_state(obs, env_id=0)

    # Cleanup
    sim.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
