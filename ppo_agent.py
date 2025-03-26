"""
Implementation of PPO agents for the block game environment.
Includes both standard PPO and MaskablePPO (with action masking).
"""

import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from agent_visualizer import visualize_agent  # Import the unified visualizer

# Only import sb3_contrib components if available (for MaskablePPO)
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

    maskable_ppo_available = True
except ImportError:
    print("Warning: sb3_contrib not found. MaskablePPO will not be available.")
    maskable_ppo_available = False

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


def train_ppo(num_envs=4, total_timesteps=100000, save_path="./models/"):
    """
    Train a standard PPO agent on the block game environment.

    Args:
        num_envs (int): Number of parallel environments to use
        total_timesteps (int): Total timesteps for training
        save_path (str): Directory to save model checkpoints

    Returns:
        The trained PPO model
    """
    # Create training vectorized environments
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # Create the PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.01,
    )

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10k steps
        save_path=save_path,
        name_prefix="ppo_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Train the model
    print("Starting training of standard PPO agent")
    start_time = time.time()

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save the final model
    final_model_path = os.path.join(save_path, "final_ppo_model")
    model.save(final_model_path)

    training_time = time.time() - start_time
    print(f"Training finished after {training_time:.2f} seconds")

    return model


def train_masked_ppo(num_envs=4, total_timesteps=100000, save_path="./models/"):
    """
    Train a PPO agent with action masking on the block game environment.

    Args:
        num_envs (int): Number of parallel environments to use
        total_timesteps (int): Total timesteps for training
        save_path (str): Directory to save model checkpoints

    Returns:
        The trained MaskablePPO model
    """
    if not maskable_ppo_available:
        print("Error: MaskablePPO is not available. Please install sb3_contrib.")
        return None

    # Create training vectorized environments - allows for parallelization
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # Create the MaskablePPO model
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.01,
    )

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10k steps
        save_path=save_path,
        name_prefix="masked_ppo_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Create evaluation callback
    eval_env = SubprocVecEnv(
        [make_env(0, seed=42)]
    )  # Use a different seed for evaluation

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=50000,
        deterministic=True,
        render=False,
    )

    # Train the model
    print("Starting training of MaskablePPO agent")
    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback]
    )

    # Save the final model
    final_model_path = os.path.join(save_path, "final_masked_ppo_model")
    model.save(final_model_path)

    training_time = time.time() - start_time
    print(f"Training finished after {training_time:.2f} seconds")

    return model


if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)

    # Set parameters directly in code
    num_envs = 8
    total_timesteps = 500000
    train_ppo_without_masking = False
    train_ppo_with_masking = False
    visualize_ppo_without_masking = True
    visualize_ppo_with_masking = False

    # Don't create the environment with render_mode="human" during training

    if train_ppo_without_masking:
        # Train standard PPO
        print(
            f"Training standard PPO with {num_envs} environments for {total_timesteps} timesteps"
        )
        model = train_ppo(
            num_envs=num_envs, total_timesteps=total_timesteps, save_path="./models/"
        )

    if train_ppo_with_masking and maskable_ppo_available:
        # Train MaskablePPO
        print(
            f"Training MaskablePPO with {num_envs} environments for {total_timesteps} timesteps"
        )
        model = train_masked_ppo(
            num_envs=num_envs, total_timesteps=total_timesteps, save_path="./models/"
        )

    # Now create environment with rendering ONLY for visualization
    env = BlockGameEnv(render_mode="human")
    env = Monitor(env)

    if visualize_ppo_without_masking:
        # Load and visualize the trained model
        print("Visualizing trained PPO agent")
        loaded_model = PPO.load("./models/final_ppo_model")
        visualize_agent(
            env,
            loaded_model,
            episodes=10,
            delay=0.2,
            use_masks=False,
            window_title="Standard PPO Agent",
        )

    if visualize_ppo_with_masking and maskable_ppo_available:
        # Load and visualize the trained model
        print("Visualizing trained MaskablePPO agent")
        loaded_model = MaskablePPO.load("./models/final_masked_ppo_model")
        visualize_agent(
            env,
            loaded_model,
            episodes=10,
            delay=0.2,
            use_masks=True,
            window_title="MaskablePPO Agent",
        )
