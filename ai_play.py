"""
Example of how to set up and train an RL agent on the block game environment.
This file shows the structure but doesn't implement a specific algorithm.
You can use this as a template for integrating with libraries like Stable Baselines3.
"""

import time
from game_env import BlockGameEnv
from random_agent import RandomAgent


def demo_train_agent(env, agent, episodes=10, max_steps=1000, delay=0.1):
    """Train an agent on the environment for a specified number of episodes."""
    total_rewards = []

    for episode in range(episodes):
        # Reset environment
        obs, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Get action from agent
            action, _ = agent.predict(obs)

            # For visualization: decode action and highlight
            if hasattr(env, "renderer") and env.renderer:
                shape_idx = action // 64
                position = action % 64
                row = position // 8
                col = position % 8
                env.renderer.set_agent_action(shape_idx, row, col)

            # Apply action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Render if in human mode
            if hasattr(env, "render"):
                env.render()
                time.sleep(delay)  # Delay for visualization

            # Check if episode is done
            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        print(
            f"Episode {episode+1}/{episodes}, Reward: {episode_reward}, Score: {info['score']}"
        )

    return total_rewards


def visualize_agent(env, agent, episodes=5, delay=0.2):
    """Visualize an agent playing the game."""
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done and step < 1000:
            # Get action from agent
            action, _ = agent.predict(obs)

            # Highlight the chosen action
            if hasattr(env, "renderer") and env.renderer:
                shape_idx = action // 64
                position = action % 64
                row = position // 8
                col = position % 8
                env.renderer.set_agent_action(shape_idx, row, col)

            # Render before taking action (to see the planned move)
            env.render()
            time.sleep(delay / 2)

            # Apply action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render after taking action
            env.render()
            time.sleep(delay / 2)

            done = terminated or truncated
            step += 1

        print(f"Episode {episode+1}: Score = {info['score']}, Reward = {total_reward}")
        time.sleep(1)  # Pause between episodes


def train_with_stable_baselines():
    """
    Example of how to use Stable Baselines3 with our environment.
    """
    # Uncomment these imports after installing stable-baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback

    print("started training with stable baselines")

    # Create and wrap the environment
    env = BlockGameEnv()
    env = DummyVecEnv([lambda: env])

    # Create the model
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/")

    # Create callback for saving model
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path="./models/", name_prefix="PPO_model"
    )

    print("Starting training")

    # Train the model
    model.learn(total_timesteps=100000, callback=checkpoint_callback)

    # Save the final model with the current date
    model.save("block_game_final_model")

    print("Training finished")

    pass


if __name__ == "__main__":
    # Create environment with rendering
    env = BlockGameEnv(render_mode="human")

    # Create a random agent
    agent = RandomAgent(env)

    # Visualize the agent playing
    print("Random Agent Playing for 5 games...")
    visualize_agent(env, agent, episodes=3, delay=0.2)

    # Example of training a random agent (just for demonstration)
    print("Training Random Agent (just for demonstration)...")
    demo_train_agent(env, agent, episodes=3, max_steps=100, delay=0.1)

    print("Training a PPO agent with StableBaselines3")
    # Example of how you would use Stable Baselines3
    train_with_stable_baselines()

    print("Looking at the PPO agent play")
    from stable_baselines3 import PPO

    model = PPO.load("block_game_final_model")
    visualize_agent(env, model, episodes=3)
