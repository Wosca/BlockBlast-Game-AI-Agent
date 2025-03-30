"""
Implementation of DQN agent for the block game environment.
"""

import time
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from agent_visualizer import visualize_agent  # Import the unified visualizer

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


def train_dqn(total_timesteps=100000, save_path="./models/", continue_training=False):
    """
    Train a DQN agent on the block game environment.

    Args:
        total_timesteps (int): Total timesteps for training
        save_path (str): Directory to save model checkpoints
        continue_training (bool): Whether to continue training from a saved model

    Returns:
        The trained DQN model
    """
    # Create training environment
    env = DummyVecEnv([make_env(0)])  # DQN typically uses a single environment

    # Check if a model already exists and if we want to continue training
    model_path = os.path.join(save_path, "final_dqn_model")
    if continue_training and os.path.exists(model_path):
        print("Loading existing model for continued training...")
        model = DQN.load(model_path, env=env)  # Load the existing model
    else:
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


if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)

    # Set parameters directly in code
    total_timesteps = 500000  # Typically DQN needs more samples than PPO
    train_dqn_agent = False
    visualize_dqn_agent = True
    continue_training = True  # Default to False

    # Don't create the environment with render_mode="human" during training
    if train_dqn_agent:
        # Train DQN
        print(f"Training DQN for {total_timesteps} timesteps")
        model = train_dqn(
            total_timesteps=total_timesteps,
            save_path="./models/",
            continue_training=continue_training,
        )

    # Now create environment with rendering ONLY for visualization
    if visualize_dqn_agent:
        env = BlockGameEnv(render_mode="human")
        env = Monitor(env)

        # Load and visualize the trained model
        print("Visualizing trained DQN agent")
        loaded_model = DQN.load("./models/final_dqn_model")
        visualize_agent(
            env,
            loaded_model,
            episodes=1,
            delay=1,
            use_masks=False,
            window_title="DQN Agent",
        )
