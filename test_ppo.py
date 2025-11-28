"""
Test trained PPO model on CartPole.

Usage:
    ~/IsaacLab/isaaclab.sh -p test_ppo.py --model ppo_cartpole_best.pth --num_envs 1 --num_episodes 5
"""

import argparse
import csv

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test PPO on CartPole")
parser.add_argument("--model", type=str, required=True, help="Path to trained model")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of test environments"
)
parser.add_argument(
    "--num_episodes", type=int, default=1, help="Number of test episodes"
)
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from cartpole_sim import CartPoleSim
from ppo import PPO


def main():
    """Test trained PPO model and save rollout data."""

    print(f"\n{'='*60}")
    print("PPO CartPole Testing")
    print(f"{'='*60}")
    print(f"Model: {args_cli.model}")
    print(f"Episodes: {args_cli.num_episodes}")
    print(f"{'='*60}\n")

    # Create environment
    sim = CartPoleSim(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        cart_mass=1.0,
        pole_mass=0.2,
        pole_length=0.5,
        gravity=9.81,
    )

    # Load trained model
    agent = PPO(state_dim=4, action_dim=1, hidden_dim=64, device=args_cli.device)
    agent.load(args_cli.model)
    agent.policy.eval()

    # Storage for all episodes
    all_episodes = []
    episode_rewards = []

    for episode in range(args_cli.num_episodes):
        # Reset environment
        obs = sim.reset()
        obs_tensor = obs["policy"].to(args_cli.device)

        # Episode storage
        episode_data = {
            "time": [],
            "cart_pos": [],
            "cart_vel": [],
            "pole_angle": [],
            "pole_vel": [],
            "action": [],
            "reward": [],
        }

        episode_reward = 0
        step = 0

        while step < args_cli.max_steps:
            with torch.no_grad():
                features = agent.policy.forward(obs_tensor)
                action = agent.policy.actor_mean(features)

            # Store data
            obs_data = obs_tensor[0].cpu().numpy()
            episode_data["time"].append(step * 0.02)  # dt = 0.02
            episode_data["cart_pos"].append(obs_data[0])
            episode_data["cart_vel"].append(obs_data[1])
            episode_data["pole_angle"].append(obs_data[2])
            episode_data["pole_vel"].append(obs_data[3])
            episode_data["action"].append(action[0].item())

            # Step
            obs, reward, terminated, truncated, info = sim.step(action)
            obs_tensor = obs["policy"].to(args_cli.device)
            done = (terminated | truncated)[0]

            episode_data["reward"].append(reward[0].item())
            episode_reward += reward[0].item()
            step += 1

            if done:
                break

        all_episodes.append(episode_data)
        episode_rewards.append(episode_reward)

        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {step}")

    # Save to CSV
    with open("ppo_rollouts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "time",
                "cart_pos",
                "cart_vel",
                "pole_angle",
                "pole_vel",
                "action",
                "reward",
            ]
        )

        for ep_idx, ep_data in enumerate(all_episodes):
            for i in range(len(ep_data["time"])):
                writer.writerow(
                    [
                        ep_idx,
                        ep_data["time"][i],
                        ep_data["cart_pos"][i],
                        ep_data["cart_vel"][i],
                        ep_data["pole_angle"][i],
                        ep_data["pole_vel"][i],
                        ep_data["action"][i],
                        ep_data["reward"][i],
                    ]
                )

    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"Episodes: {len(episode_rewards)}")
    print(
        f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"\nData saved to: ppo_rollouts.csv")
    print(f"{'='*60}\n")

    sim.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
