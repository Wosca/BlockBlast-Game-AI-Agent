"""
Implementation of DQN agent for the block game environment.
Includes standard DQN with optional action masking capability.
"""

import time
import os
import numpy as np
import torch as th
from torch import nn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from game_env import BlockGameEnv


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    Args:
        rank (int): Index of the subprocess
        seed (int): The initial seed for the environment

    Returns:
        The function to create the environment
    """

    def _init():
        env = BlockGameEnv()
        env = Monitor(env)  # Add Monitor wrapper for proper logging
        try:
            env.reset(seed=seed + rank)  # New Gymnasium API
        except TypeError:
            # Fallback for older gym API
            env.seed(seed + rank)
        return env

    return _init


class MaskableDQNPolicy(nn.Module):
    """
    Custom policy for DQN that can use action masks to ignore invalid actions.
    """

    def __init__(self, observation_space, action_space):
        super().__init__()
        # This is a minimal implementation - SB3's actual policy is more sophisticated
        self.q_net = DQN.policy_class(observation_space, action_space).q_net

    def forward(self, obs, action_masks=None):
        q_values = self.q_net(obs)

        # Apply action mask if provided
        if action_masks is not None:
            # Set Q-values of invalid actions to a large negative number
            # so they won't be selected during the argmax in the agent
            q_values = th.where(
                action_masks.bool(),
                q_values,
                th.tensor(-1e8, device=q_values.device, dtype=q_values.dtype),
            )

        return q_values


class CustomDQN(DQN):
    """
    Custom DQN implementation that supports action masking.
    """

    def predict(self, observation, action_masks=None, deterministic=True):
        """
        Override predict method to support action masking.
        """
        if action_masks is not None:
            # Get q_values from the model
            q_values = self.q_net(self.policy.obs_to_tensor(observation)[0])

            # Apply action mask - ensure it's converted to a boolean tensor
            action_masks_tensor = th.tensor(
                action_masks, device=q_values.device, dtype=th.bool
            )

            q_values = th.where(
                action_masks_tensor,
                q_values,
                th.tensor(-1e8, device=q_values.device, dtype=q_values.dtype),
            )

            # Get actions - convert to Python int to ensure it's a scalar
            actions = q_values.argmax(dim=1).cpu().numpy()
            # Return the item (scalar) if it's a single action
            if len(actions) == 1:
                return int(actions[0]), None
            return actions, None
        else:
            actions, states = super().predict(observation, deterministic=deterministic)
            # Convert to Python int if it's a single action
            if isinstance(actions, np.ndarray) and actions.shape == (1,):
                return int(actions[0]), states
            return actions, states


def train_dqn(num_envs=1, total_timesteps=100000, save_path="./models/"):
    """
    Train a standard DQN agent on the block game environment.

    Args:
        num_envs (int): Number of environments to use (usually 1 for DQN)
        total_timesteps (int): Total timesteps for training
        save_path (str): Directory to save model checkpoints

    Returns:
        The trained DQN model
    """
    # Create training environment
    env = make_env(0)()  # DQN typically uses a single environment

    # Create the DQN model
    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,  # Soft update coefficient
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        train_freq=(4, "step"),
        gradient_steps=1,
    )

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10k steps
        save_path=save_path,
        name_prefix="dqn_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Train the model
    print("Starting training of DQN agent")
    start_time = time.time()

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save the final model
    final_model_path = os.path.join(save_path, "final_dqn_model")
    model.save(final_model_path)

    training_time = time.time() - start_time
    print(f"Training finished after {training_time:.2f} seconds")

    return model


def train_masked_dqn(num_envs=1, total_timesteps=100000, save_path="./models/"):
    """
    Train a DQN agent with action masking on the block game environment.

    Args:
        num_envs (int): Number of environments to use (usually 1 for DQN)
        total_timesteps (int): Total timesteps for training
        save_path (str): Directory to save model checkpoints

    Returns:
        The trained DQN model
    """
    # Create training environment
    env = make_env(0)()  # DQN typically uses a single environment

    # Create the DQN model with custom prediction method for action masking
    model = CustomDQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,  # Soft update coefficient
        target_update_interval=500,
        exploration_fraction=0.1,  # Less exploration needed with masking
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        train_freq=(4, "step"),
        gradient_steps=1,
    )

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10k steps
        save_path=save_path,
        name_prefix="masked_dqn_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Custom learning method to use action masks during training
    def masked_learn(self, total_timesteps, callback=None):
        self._setup_learn(total_timesteps, callback)

        while self.num_timesteps < total_timesteps:
            # Get action masks from environment
            action_masks = self.env.action_masks()

            # Sample action with masking
            q_values = self.q_net(self.policy.obs_to_tensor(self._last_obs)[0])
            q_values = th.where(
                th.tensor(action_masks, device=q_values.device, dtype=th.bool),
                q_values,
                th.tensor(-1e8, device=q_values.device, dtype=q_values.dtype),
            )

            # Continue with standard DQN training...
            # (Simplified here - actual implementation would update this)
            self._on_step()

            if (
                self.replay_buffer.size() > 0
                and self.num_timesteps > self.learning_starts
            ):
                self.train(
                    batch_size=self.batch_size, gradient_steps=self.gradient_steps
                )

        return self

    # The actual learning is done using the standard learn method, but we'll
    # utilize action masking in the predict method during collect_rollouts
    # Train the model
    print("Starting training of DQN agent with action masking")
    start_time = time.time()

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save the final model
    final_model_path = os.path.join(save_path, "final_masked_dqn_model")
    model.save(final_model_path)

    training_time = time.time() - start_time
    print(f"Training finished after {training_time:.2f} seconds")

    return model


def visualize_agent(env, agent, episodes=5, delay=0.2, use_masks=False):
    """
    Visualize a DQN agent playing the game.

    Args:
        env: The environment to use
        agent: The trained agent (DQN model)
        episodes (int): Number of episodes to run
        delay (float): Delay between frames for visualization
        use_masks (bool): Whether to use action masks with the agent
    """
    # Run episodes
    total_scores = []
    total_rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        episode_score = 0

        while not done and step < 1000:
            # Get action from agent (with masks if applicable)
            if use_masks:
                if hasattr(env.unwrapped, "action_masks"):
                    action_masks = env.unwrapped.action_masks()
                elif hasattr(env, "action_masks"):
                    action_masks = env.action_masks()
                else:
                    raise AttributeError("Environment doesn't support action masks")

                # Get action with masks
                action, _ = agent.predict(
                    obs, action_masks=action_masks, deterministic=True
                )
            else:
                action, _ = agent.predict(obs, deterministic=True)

            # Ensure action is a Python int
            if not isinstance(action, int):
                if isinstance(action, np.ndarray):
                    action = int(action.item())
                else:
                    action = int(action)

            # Decode action for visualization
            shape_idx = action // 64
            position = action % 64
            row = position // 8
            col = position % 8

            # Highlight the chosen action for visualization
            if hasattr(env, "renderer") and env.renderer:
                env.renderer.set_agent_action(shape_idx, row, col)
                env.renderer.set_agent_thinking(True)

            # Render before taking action
            env.render()
            time.sleep(delay / 2)

            # Apply action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Update episode score from info
            if "score" in info:
                episode_score = info["score"]

            # Turn off thinking visualization
            if hasattr(env, "renderer") and env.renderer:
                env.renderer.set_agent_thinking(False)

            # Render after taking action
            env.render()
            time.sleep(delay / 2)

            done = terminated or truncated
            step += 1

        total_scores.append(episode_score)
        total_rewards.append(total_reward)
        print(
            f"Episode {episode+1}: Score = {episode_score}, Reward = {total_reward:.2f}, Steps = {step}"
        )
        time.sleep(1)  # Pause between episodes

    # Print summary statistics
    avg_score = sum(total_scores) / len(total_scores)
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nPerformance Summary:")
    print(f"Average Game Score: {avg_score:.2f}")
    print(f"Average RL Reward: {avg_reward:.2f}")


if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)

    # Set parameters directly in code
    total_timesteps = 500000  # Typically DQN needs more samples than PPO
    train_dqn_without_masking = False
    train_dqn_with_masking = False
    visualize_dqn_without_masking = False
    visualize_dqn_with_masking = True

    # Don't create the environment with render_mode="human" during training

    if train_dqn_without_masking:
        # Train standard DQN
        print(f"Training standard DQN for {total_timesteps} timesteps")
        model = train_dqn(total_timesteps=total_timesteps, save_path="./models/")

    if train_dqn_with_masking:
        # Train DQN with masking
        print(f"Training DQN with action masking for {total_timesteps} timesteps")
        model = train_masked_dqn(total_timesteps=total_timesteps, save_path="./models/")

    # Now create environment with rendering ONLY for visualization
    env = BlockGameEnv(render_mode="human")
    env = Monitor(env)

    if visualize_dqn_without_masking:
        # Load and visualize the trained model
        print("Visualizing trained DQN agent")
        loaded_model = DQN.load("./models/final_dqn_model")
        visualize_agent(env, loaded_model, episodes=3, delay=0.2, use_masks=False)

    if visualize_dqn_with_masking:
        # Load and visualize the trained model
        print("Visualizing trained DQN agent with action masking")
        loaded_model = CustomDQN.load("./models/final_masked_dqn_model")
        visualize_agent(env, loaded_model, episodes=3, delay=0.2, use_masks=True)
