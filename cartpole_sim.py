import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import (
    CartpoleEnvCfg,
)


class CartPoleSim:
    """CartPole simulation wrapper with custom reward function."""

    def __init__(
        self,
        num_envs=16,
        device="cuda:0",
        cart_mass=1.0,
        pole_mass=0.2,
        pole_length=0.5,
        gravity=9.81,
        use_custom_reward=True,
    ):
        """
        Initialize CartPole simulation.

        Args:
            num_envs: Number of parallel environments
            device: Computation device
            cart_mass: Cart mass (kg)
            pole_mass: Pole mass (kg)
            pole_length: Pole length from pivot to COM (m)
            gravity: Gravitational acceleration (m/s²)
            use_custom_reward: Use custom reward function instead of default
        """
        self.num_envs = num_envs
        self.device = device
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length
        self.gravity = gravity
        self.use_custom_reward = use_custom_reward

        self.env = None
        self.step_count = 0

        # Reward weights (tunable)
        self.reward_weights = {
            "pole_upright": 1.0,  # Reward for keeping pole upright
            "cart_centered": 0.5,  # Reward for keeping cart near center
            "low_velocity": 0.1,  # Reward for low velocities
            "action_penalty": 0.01,  # Penalty for large actions
            "alive_bonus": 1.0,  # Bonus for staying alive
        }

        self._setup_environment()
        self._set_custom_parameters()

    def _setup_environment(self):
        """Create and configure the environment."""
        env_cfg = CartpoleEnvCfg()
        env_cfg.scene.num_envs = self.num_envs
        env_cfg.sim.device = self.device
        env_cfg.sim.gravity = (0.0, 0.0, -self.gravity)

        self.env = ManagerBasedRLEnv(cfg=env_cfg)
        print(f"[INFO] Environment created with {self.num_envs} environments")

    def _set_custom_parameters(self):
        """Override masses with custom values."""
        print(f"\n[INFO] Setting custom physical parameters:")
        print(f"  Cart mass: {self.cart_mass} kg")
        print(f"  Pole mass: {self.pole_mass} kg")
        print(f"  Pole length: {self.pole_length} m")
        print(f"  Gravity: {self.gravity} m/s²")
        print(
            f"  Custom rewards: {'Enabled' if self.use_custom_reward else 'Disabled'}\n"
        )

        robot = self.env.scene["robot"]

        cart_body_idx = robot.find_bodies("cart")[0][0]
        pole_body_idx = robot.find_bodies("pole")[0][0]

        masses = robot.root_physx_view.get_masses()
        print(f"[INFO] Original cart mass: {masses[0, cart_body_idx].item():.3f} kg")
        print(f"[INFO] Original pole mass: {masses[0, pole_body_idx].item():.3f} kg")

        masses[:, cart_body_idx] = self.cart_mass
        masses[:, pole_body_idx] = self.pole_mass

        env_ids = torch.arange(self.env.num_envs, dtype=torch.int32, device="cpu")
        robot.root_physx_view.set_masses(masses, env_ids)

        print(f"[INFO] Updated cart mass: {self.cart_mass} kg")
        print(f"[INFO] Updated pole mass: {self.pole_mass} kg\n")

    def compute_custom_reward(self, obs, actions):
        """
        Compute custom reward based on state and actions.

        Reward components:
        1. Pole upright: Exponential reward for keeping pole vertical
        2. Cart centered: Gaussian reward for staying near origin
        3. Low velocity: Penalty for high velocities (smoother control)
        4. Action penalty: Penalty for large control efforts
        5. Alive bonus: Constant reward for not terminating

        Args:
            obs: Observation dictionary with 'policy' key
            actions: Action tensor (num_envs, action_dim)

        Returns:
            reward: Tensor of shape (num_envs,)
        """
        # Extract state components
        obs_data = obs["policy"]  # Shape: (num_envs, 4)
        cart_pos = obs_data[:, 0]  # Cart position
        cart_vel = obs_data[:, 1]  # Cart velocity
        pole_angle = obs_data[:, 2]  # Pole angle from vertical (rad)
        pole_vel = obs_data[:, 3]  # Pole angular velocity

        # 1. Pole upright reward (exponential - highest when upright)
        # Range: [0, 1], max at angle=0
        pole_upright_reward = torch.exp(-10.0 * pole_angle**2)

        # 2. Cart centered reward (Gaussian - highest at center)
        # Range: [0, 1], max at position=0
        cart_centered_reward = torch.exp(-0.5 * cart_pos**2)

        # 3. Low velocity reward (penalty for high speeds)
        # Encourage smooth, controlled motion
        velocity_penalty = -(cart_vel**2 + pole_vel**2)

        # 4. Action penalty (discourage large control efforts)
        action_penalty = -torch.sum(actions**2, dim=-1)

        # 5. Alive bonus (constant reward for each step alive)
        alive_bonus = torch.ones(self.num_envs, device=self.device)

        # Combine weighted rewards
        total_reward = (
            self.reward_weights["pole_upright"] * pole_upright_reward
            + self.reward_weights["cart_centered"] * cart_centered_reward
            + self.reward_weights["low_velocity"] * velocity_penalty
            + self.reward_weights["action_penalty"] * action_penalty
            + self.reward_weights["alive_bonus"] * alive_bonus
        )

        return total_reward

    def set_reward_weights(self, **kwargs):
        """
        Update reward weights.

        Example:
            sim.set_reward_weights(pole_upright=2.0, cart_centered=0.3)
        """
        for key, value in kwargs.items():
            if key in self.reward_weights:
                self.reward_weights[key] = value
                print(f"[INFO] Updated reward weight '{key}': {value}")
            else:
                print(f"[WARNING] Unknown reward weight: {key}")

    def reset(self):
        """Reset the environment."""
        self.step_count = 0
        obs, _ = self.env.reset()
        print("-" * 80)
        print("[INFO] Environment reset")
        return obs

    def step(self, actions):
        """
        Step the environment with optional custom rewards.

        Args:
            actions: Action tensor (num_envs, action_dim)

        Returns:
            observations, rewards, terminated, truncated, info
        """
        # Step the base environment
        obs, reward_default, terminated, truncated, info = self.env.step(actions)

        # Override with custom reward if enabled
        if self.use_custom_reward:
            reward = self.compute_custom_reward(obs, actions)
        else:
            reward = reward_default

        self.step_count += 1
        return obs, reward, terminated, truncated, info

    def get_random_actions(self):
        """Generate random actions for all environments."""
        return torch.randn_like(self.env.action_manager.action)

    def print_state(self, obs, env_id=0):
        """Print current state of specified environment."""
        obs_data = obs["policy"][env_id]

        cart_pos = obs_data[0].item()
        cart_vel = obs_data[1].item()
        pole_angle = obs_data[2].item()
        pole_vel = obs_data[3].item()

        print(
            f"[Env {env_id}] Step {self.step_count}: "
            f"x={cart_pos:6.3f}, vx={cart_vel:6.3f}, "
            f"θ={pole_angle:6.3f}, vθ={pole_vel:6.3f}"
        )

    def get_reward_info(self, obs, actions):
        """
        Get detailed reward breakdown for analysis.

        Returns:
            Dictionary with individual reward components
        """
        obs_data = obs["policy"]
        cart_pos = obs_data[:, 0]
        cart_vel = obs_data[:, 1]
        pole_angle = obs_data[:, 2]
        pole_vel = obs_data[:, 3]

        return {
            "pole_upright": torch.exp(-10.0 * pole_angle**2).mean().item(),
            "cart_centered": torch.exp(-0.5 * cart_pos**2).mean().item(),
            "velocity_penalty": -(cart_vel**2 + pole_vel**2).mean().item(),
            "action_penalty": -torch.sum(actions**2, dim=-1).mean().item(),
            "alive_bonus": 1.0,
        }

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()
            print("\n[INFO] Environment closed")
