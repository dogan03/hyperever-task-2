"""
Train PPO on CartPole environment with TensorBoard logging.

Usage:
    ~/IsaacLab/isaaclab.sh -p train_ppo.py --num_envs 4096 --headless

View training in TensorBoard:
    tensorboard --logdir=runs
"""

import argparse
from datetime import datetime

import numpy as np
import torch
from isaaclab.app import AppLauncher
from torch.utils.tensorboard import SummaryWriter

# Parse arguments
parser = argparse.ArgumentParser(description="Train PPO on CartPole with TensorBoard")
parser.add_argument(
    "--num_envs", type=int, default=4096, help="Number of parallel environments"
)
parser.add_argument(
    "--max_iterations", type=int, default=500, help="Maximum training iterations"
)
parser.add_argument(
    "--steps_per_iteration", type=int, default=32, help="Steps per iteration"
)
parser.add_argument(
    "--save_interval", type=int, default=50, help="Save model every N iterations"
)
parser.add_argument(
    "--log_dir", type=str, default="runs", help="TensorBoard log directory"
)
parser.add_argument(
    "--run_name", type=str, default=None, help="Name for this training run"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching
from cartpole_sim import CartPoleSim
from ppo import PPO


class TrainingLogger:
    """Handles TensorBoard logging and statistics tracking."""

    def __init__(self, log_dir, run_name=None):
        # Create run name with timestamp
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"ppo_cartpole_{timestamp}"

        self.writer = SummaryWriter(f"{log_dir}/{run_name}")
        self.episode_rewards = []
        self.episode_lengths = []

        print(f"[INFO] TensorBoard logging to: {log_dir}/{run_name}")
        print(f"[INFO] Run: tensorboard --logdir={log_dir}")

    def log_hyperparameters(self, hparams):
        """Log hyperparameters to TensorBoard."""
        # Convert to string format for display
        hparam_dict = {k: v for k, v in hparams.items()}
        self.writer.add_hparams(
            hparam_dict, {"hparam/placeholder": 0}  # Required but unused
        )

    def log_iteration(self, iteration, metrics):
        """Log metrics for current iteration."""
        step = metrics["total_steps"]

        # Training metrics
        self.writer.add_scalar("Loss/Policy", metrics["policy_loss"], step)
        self.writer.add_scalar("Loss/Value", metrics["value_loss"], step)
        self.writer.add_scalar("Loss/Entropy", metrics["entropy"], step)
        self.writer.add_scalar("Loss/Total", metrics.get("total_loss", 0), step)

        # Reward metrics
        self.writer.add_scalar("Reward/Mean", metrics["mean_reward"], step)
        self.writer.add_scalar("Reward/Std", metrics["std_reward"], step)
        self.writer.add_scalar("Reward/Min", metrics["min_reward"], step)
        self.writer.add_scalar("Reward/Max", metrics["max_reward"], step)

        # Episode statistics
        if "episode_length" in metrics:
            self.writer.add_scalar("Episode/Length", metrics["episode_length"], step)

        # Value function statistics
        if "mean_value" in metrics:
            self.writer.add_scalar("Value/Mean", metrics["mean_value"], step)
            self.writer.add_scalar("Value/Std", metrics["value_std"], step)

        # Advantage statistics
        if "mean_advantage" in metrics:
            self.writer.add_scalar("Advantage/Mean", metrics["mean_advantage"], step)
            self.writer.add_scalar("Advantage/Std", metrics["advantage_std"], step)

        # Learning rate
        if "learning_rate" in metrics:
            self.writer.add_scalar("Train/LearningRate", metrics["learning_rate"], step)

        # Iteration counter
        self.writer.add_scalar("Train/Iteration", iteration, step)

        # Track episode rewards
        self.episode_rewards.append(metrics["mean_reward"])
        if len(self.episode_rewards) >= 10:
            recent_mean = np.mean(self.episode_rewards[-10:])
            self.writer.add_scalar("Reward/Recent10", recent_mean, step)
        if len(self.episode_rewards) >= 100:
            recent_mean = np.mean(self.episode_rewards[-100:])
            self.writer.add_scalar("Reward/Recent100", recent_mean, step)

    def log_text(self, tag, text, step):
        """Log text to TensorBoard."""
        self.writer.add_text(tag, text, step)

    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()
        print("[INFO] TensorBoard writer closed")


def main():
    """Main training loop."""

    # Hyperparameters
    CART_MASS = 1.0
    POLE_MASS = 0.2
    POLE_LENGTH = 0.5
    GRAVITY = 9.81
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2

    # Create TensorBoard logger
    logger = TrainingLogger(args_cli.log_dir, args_cli.run_name)

    # Log hyperparameters
    hyperparams = {
        "num_envs": args_cli.num_envs,
        "max_iterations": args_cli.max_iterations,
        "steps_per_iteration": args_cli.steps_per_iteration,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_epsilon": CLIP_EPSILON,
        "cart_mass": CART_MASS,
        "pole_mass": POLE_MASS,
        "pole_length": POLE_LENGTH,
        "gravity": GRAVITY,
    }
    logger.log_hyperparameters(hyperparams)

    # Create environment
    print("[INFO] Creating environment...")
    sim = CartPoleSim(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        cart_mass=CART_MASS,
        pole_mass=POLE_MASS,
        pole_length=POLE_LENGTH,
        gravity=GRAVITY,
    )

    # Create PPO agent
    print("[INFO] Creating PPO agent...")
    agent = PPO(
        state_dim=4,
        action_dim=1,
        hidden_dim=64,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_epsilon=CLIP_EPSILON,
        device=args_cli.device,
    )

    # Training statistics
    best_reward = -float("inf")

    # Reset environment
    obs = sim.reset()
    obs_tensor = obs["policy"].to(args_cli.device)

    print(f"\n{'='*80}")
    print(f"Starting PPO Training")
    print(f"{'='*80}")
    print(f"Environments: {args_cli.num_envs}")
    print(f"Max iterations: {args_cli.max_iterations}")
    print(f"Steps per iteration: {args_cli.steps_per_iteration}")
    print(f"TensorBoard: tensorboard --logdir={args_cli.log_dir}")
    print(f"{'='*80}\n")

    # Training loop
    for iteration in range(args_cli.max_iterations):
        # Storage for rollout
        states_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        dones_list = []
        values_list = []

        # Collect rollout
        for step in range(args_cli.steps_per_iteration):
            # Select action
            with torch.no_grad():
                action, log_prob = agent.select_action(obs_tensor)
                value = agent.policy.get_value(obs_tensor)

            # Store transition
            states_list.append(obs_tensor)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            values_list.append(value)

            # Step environment
            obs, reward, terminated, truncated, info = sim.step(action)
            obs_tensor = obs["policy"].to(args_cli.device)
            done = terminated | truncated

            # Store reward and done
            rewards_list.append(reward)
            dones_list.append(done.float())

        # Get final value for GAE
        with torch.no_grad():
            next_value = agent.policy.get_value(obs_tensor)

        # Convert lists to tensors
        states = torch.stack(states_list)
        actions = torch.stack(actions_list)
        log_probs = torch.stack(log_probs_list)
        rewards = torch.stack(rewards_list)
        dones = torch.stack(dones_list)
        values = torch.stack(values_list)

        # Compute advantages and returns
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

        # Flatten for update (combine time and environment dimensions)
        states_flat = states.view(-1, 4)
        actions_flat = actions.view(-1, 1)
        log_probs_flat = log_probs.view(-1)
        returns_flat = returns.view(-1)
        advantages_flat = advantages.view(-1)

        # Update policy
        losses = agent.update(
            states_flat,
            actions_flat,
            log_probs_flat,
            returns_flat,
            advantages_flat,
            n_epochs=4,
            batch_size=2048,
        )

        # Update statistics
        agent.total_steps += args_cli.num_envs * args_cli.steps_per_iteration

        # Compute metrics
        mean_reward = rewards.mean().item()
        std_reward = rewards.std().item()
        min_reward = rewards.min().item()
        max_reward = rewards.max().item()
        mean_value = values.mean().item()
        value_std = values.std().item()
        mean_advantage = advantages.mean().item()
        advantage_std = advantages.std().item()

        # Log to TensorBoard
        metrics = {
            "total_steps": agent.total_steps,
            "policy_loss": losses["policy_loss"],
            "value_loss": losses["value_loss"],
            "entropy": losses["entropy"],
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "mean_value": mean_value,
            "value_std": value_std,
            "mean_advantage": mean_advantage,
            "advantage_std": advantage_std,
            "learning_rate": LEARNING_RATE,
        }
        logger.log_iteration(iteration, metrics)

        # Print progress
        if iteration % 10 == 0:
            recent_rewards = (
                np.mean(logger.episode_rewards[-10:])
                if len(logger.episode_rewards) >= 10
                else mean_reward
            )
            print(
                f"Iter {iteration:4d} | "
                f"Steps: {agent.total_steps:8d} | "
                f"Reward: {recent_rewards:7.2f} | "
                f"PL: {losses['policy_loss']:6.4f} | "
                f"VL: {losses['value_loss']:6.4f} | "
                f"Ent: {losses['entropy']:6.4f}"
            )

        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            agent.save("ppo_cartpole_best.pth")
            logger.log_text(
                "Model",
                f"New best model at iteration {iteration} with reward {best_reward:.2f}",
                iteration,
            )

        # Save checkpoint
        if iteration % args_cli.save_interval == 0 and iteration > 0:
            agent.save(f"ppo_cartpole_iter_{iteration}.pth")
            logger.log_text(
                "Checkpoint", f"Saved checkpoint at iteration {iteration}", iteration
            )

    # Final save
    agent.save("ppo_cartpole_final.pth")

    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"Total steps: {agent.total_steps}")
    print(f"Best reward: {best_reward:.2f}")
    print(f"{'='*80}\n")

    # Cleanup
    logger.close()
    sim.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
