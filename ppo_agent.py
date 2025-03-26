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


def visualize_agent(env, agent, episodes=5, delay=0.2, use_masks=False):
    """
    Visualize an agent (either PPO or MaskablePPO) playing the game.

    Args:
        env: The environment to use
        agent: The trained agent (PPO or MaskablePPO model)
        episodes (int): Number of episodes to run
        delay (float): Delay between frames for visualization
        use_masks (bool): Whether to use action masks with the agent
    """
    # Get action masks function if needed and available
    get_action_masks_fn = None
    if use_masks and maskable_ppo_available:
        from sb3_contrib.common.maskable.utils import get_action_masks

        get_action_masks_fn = get_action_masks

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
            if use_masks and get_action_masks_fn is not None:
                if hasattr(env.unwrapped, "action_masks"):
                    action_masks = env.unwrapped.action_masks()
                elif hasattr(env, "action_masks"):
                    action_masks = env.action_masks()
                else:
                    raise AttributeError("Environment doesn't support action masks")
                action, _ = agent.predict(
                    obs, action_masks=action_masks, deterministic=True
                )
            else:
                action, _ = agent.predict(obs, deterministic=True)

            # Highlight the chosen action for visualization
            if hasattr(env, "renderer") and env.renderer:
                shape_idx = action // 64
                position = action % 64
                row = position // 8
                col = position % 8
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
    num_envs = 6
    total_timesteps = 1000000
    train_ppo_without_masking = False
    train_ppo_with_masking = False
    visualize_ppo_without_masking = False
    visualize_ppo_with_masking = True

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
        visualize_agent(env, loaded_model, episodes=3, delay=0.2, use_masks=False)

    if visualize_ppo_with_masking and maskable_ppo_available:
        # Load and visualize the trained model
        print("Visualizing trained MaskablePPO agent")
        loaded_model = MaskablePPO.load("./models/final_masked_ppo_model")
        visualize_agent(env, loaded_model, episodes=3, delay=0.2, use_masks=True)
