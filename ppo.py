import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, state_dim=4, action_dim=1, hidden_dim=64):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        features = self.shared(state)
        return features

    def act(self, state):
        """Sample action from policy."""
        features = self.forward(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)

        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def evaluate(self, state, action):
        """Evaluate actions for PPO update."""
        features = self.forward(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)

        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)

        return log_prob, value, entropy

    def get_value(self, state):
        """Get state value."""
        features = self.forward(state)
        return self.critic(features).squeeze(-1)


class PPO:
    """Proximal Policy Optimization algorithm."""

    def __init__(
        self,
        state_dim=4,
        action_dim=1,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        vf_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device="cuda:0",
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Create actor-critic network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Statistics
        self.total_steps = 0

    def select_action(self, state):
        """Select action using current policy."""
        with torch.no_grad():
            action, log_prob = self.policy.act(state)
        return action, log_prob

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            )

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        states,
        actions,
        log_probs_old,
        returns,
        advantages,
        n_epochs=4,
        batch_size=64,
    ):
        """Update policy using PPO."""
        num_samples = states.size(0)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for _ in range(n_epochs):
            # Generate random indices for mini-batches
            indices = torch.randperm(num_samples)

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate(
                    batch_states, batch_actions
                )

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # Return average losses
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path):
        """Save model."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            path,
        )
        print(f"[INFO] Model saved to {path}")

    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        print(f"[INFO] Model loaded from {path}")
