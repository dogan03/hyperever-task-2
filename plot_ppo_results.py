"""
Plot PPO evaluation results for assignment submission.

Usage:
    python plot_ppo_results.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv("ppo_rollouts.csv")

# Create figure with subplots
fig = plt.figure(figsize=(14, 10))

# Get unique episodes
episodes = data["episode"].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(episodes)))

# Plot 1: Pole Angle over Time
ax1 = plt.subplot(3, 2, 1)
for i, ep in enumerate(episodes):
    ep_data = data[data["episode"] == ep]
    ax1.plot(
        ep_data["time"],
        ep_data["pole_angle"] * 180 / np.pi,
        color=colors[i],
        label=f"Episode {ep+1}",
        alpha=0.8,
    )
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Pole Angle (degrees)")
ax1.set_title("Pole Angle Trajectories")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)
ax1.axhline(y=0, color="r", linestyle="--", linewidth=1, alpha=0.5)

# Plot 2: Cart Position over Time
ax2 = plt.subplot(3, 2, 2)
for i, ep in enumerate(episodes):
    ep_data = data[data["episode"] == ep]
    ax2.plot(
        ep_data["time"],
        ep_data["cart_pos"],
        color=colors[i],
        label=f"Episode {ep+1}",
        alpha=0.8,
    )
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Cart Position (m)")
ax2.set_title("Cart Position Trajectories")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)
ax2.axhline(y=0, color="r", linestyle="--", linewidth=1, alpha=0.5)

# Plot 3: Control Actions over Time
ax3 = plt.subplot(3, 2, 3)
for i, ep in enumerate(episodes):
    ep_data = data[data["episode"] == ep]
    ax3.plot(
        ep_data["time"],
        ep_data["action"],
        color=colors[i],
        label=f"Episode {ep+1}",
        alpha=0.8,
    )
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Control Force (N)")
ax3.set_title("Control Actions")
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=8)

# Plot 4: Cart Velocity over Time
ax4 = plt.subplot(3, 2, 4)
for i, ep in enumerate(episodes):
    ep_data = data[data["episode"] == ep]
    ax4.plot(
        ep_data["time"],
        ep_data["cart_vel"],
        color=colors[i],
        label=f"Episode {ep+1}",
        alpha=0.8,
    )
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Cart Velocity (m/s)")
ax4.set_title("Cart Velocity")
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=8)

# Plot 5: Pole Angular Velocity over Time
ax5 = plt.subplot(3, 2, 5)
for i, ep in enumerate(episodes):
    ep_data = data[data["episode"] == ep]
    ax5.plot(
        ep_data["time"],
        ep_data["pole_vel"],
        color=colors[i],
        label=f"Episode {ep+1}",
        alpha=0.8,
    )
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Pole Angular Velocity (rad/s)")
ax5.set_title("Pole Angular Velocity")
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=8)

# Plot 6: Rewards over Time
ax6 = plt.subplot(3, 2, 6)
for i, ep in enumerate(episodes):
    ep_data = data[data["episode"] == ep]
    cumulative_reward = np.cumsum(ep_data["reward"])
    ax6.plot(
        ep_data["time"],
        cumulative_reward,
        color=colors[i],
        label=f"Episode {ep+1}",
        alpha=0.8,
    )
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Cumulative Reward")
ax6.set_title("Cumulative Rewards")
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=8)

plt.tight_layout()
plt.savefig("ppo_evaluation_rollouts.png", dpi=300, bbox_inches="tight")
print("✓ Saved: ppo_evaluation_rollouts.png")

# Create summary statistics plot
fig2, axes = plt.subplots(2, 2, figsize=(12, 8))

# Episode rewards
episode_rewards = []
for ep in episodes:
    ep_data = data[data["episode"] == ep]
    episode_rewards.append(ep_data["reward"].sum())

axes[0, 0].bar(
    range(1, len(episode_rewards) + 1), episode_rewards, color="steelblue", alpha=0.7
)
axes[0, 0].axhline(
    y=np.mean(episode_rewards),
    color="r",
    linestyle="--",
    label=f"Mean: {np.mean(episode_rewards):.1f}",
)
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Total Reward")
axes[0, 0].set_title("Episode Rewards")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Episode lengths
episode_lengths = []
for ep in episodes:
    ep_data = data[data["episode"] == ep]
    episode_lengths.append(len(ep_data))

axes[0, 1].bar(
    range(1, len(episode_lengths) + 1), episode_lengths, color="darkorange", alpha=0.7
)
axes[0, 1].axhline(
    y=np.mean(episode_lengths),
    color="r",
    linestyle="--",
    label=f"Mean: {np.mean(episode_lengths):.1f}",
)
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Steps")
axes[0, 1].set_title("Episode Lengths")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Final pole angles
final_angles = []
for ep in episodes:
    ep_data = data[data["episode"] == ep]
    final_angles.append(ep_data["pole_angle"].iloc[-1] * 180 / np.pi)

axes[1, 0].bar(range(1, len(final_angles) + 1), final_angles, color="green", alpha=0.7)
axes[1, 0].axhline(y=0, color="r", linestyle="--", linewidth=1)
axes[1, 0].set_xlabel("Episode")
axes[1, 0].set_ylabel("Final Angle (degrees)")
axes[1, 0].set_title("Final Pole Angles")
axes[1, 0].grid(True, alpha=0.3)

# Summary text
summary_text = f"""
Performance Summary:

Episodes: {len(episode_rewards)}
Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}
Mean Length: {np.mean(episode_lengths):.1f} steps
Final Angle: {np.mean([abs(a) for a in final_angles]):.2f}° (avg absolute)

Success: All episodes completed
"""

axes[1, 1].text(
    0.1,
    0.5,
    summary_text,
    fontsize=11,
    verticalalignment="center",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("ppo_performance_summary.png", dpi=300, bbox_inches="tight")
print("✓ Saved: ppo_performance_summary.png")

plt.show()

print("\n" + "=" * 60)
print("Plotting Complete!")
print("=" * 60)
print("Generated files:")
print("  1. ppo_evaluation_rollouts.png  - State/action trajectories")
print("  2. ppo_performance_summary.png  - Performance metrics")
print("=" * 60)
