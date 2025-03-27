"""
Utility for visualizing agents playing the block game.
Provides a consistent interface for visualizing different types of agents.
"""

import time
import numpy as np


def visualize_agent(
    env, agent, episodes=5, delay=0.2, use_masks=False, window_title=None
):
    """
    Visualize an agent playing the block game.

    Args:
        env: The environment to use
        agent: The trained agent (any model with a predict method)
        episodes (int): Number of episodes to run
        delay (float): Delay between frames for visualization
        use_masks (bool): Whether to use action masks with the agent
        window_title (str): Optional title for the pygame window
    """
    # Set the window title if provided
    if window_title and hasattr(env, "set_window_title"):
        env.set_window_title(window_title)
    elif window_title and hasattr(env.unwrapped, "set_window_title"):
        env.unwrapped.set_window_title(window_title)
    elif (
        window_title
        and hasattr(env, "unwrapped")
        and hasattr(env.unwrapped, "renderer")
        and env.unwrapped.renderer
    ):
        if hasattr(env.unwrapped.renderer, "set_window_title"):
            env.unwrapped.renderer.set_window_title(window_title)

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

                # Print valid/invalid actions
                valid_count = np.sum(action_masks)
                total_count = len(action_masks)
                print(f"Valid actions: {valid_count}/{total_count}")

                action, _ = agent.predict(
                    obs, action_masks=action_masks, deterministic=True
                )
            else:
                action, _ = agent.predict(obs, deterministic=True)

            # Decode action for visualization
            shape_idx = action // 64
            position = action % 64
            row = position // 8
            col = position % 8
            print(
                f"Selected action: {action} (Shape {shape_idx}, Row {row}, Col {col})"
            )

            # Highlight the chosen action for visualization if renderer is available
            if hasattr(env, "renderer") and env.renderer:
                env.renderer.set_agent_action(shape_idx, row, col)
                env.renderer.set_agent_thinking(True)
            elif (
                hasattr(env, "unwrapped")
                and hasattr(env.unwrapped, "renderer")
                and env.unwrapped.renderer
            ):
                env.unwrapped.renderer.set_agent_action(shape_idx, row, col)
                env.unwrapped.renderer.set_agent_thinking(True)

            # Render before taking action
            env.render()
            time.sleep(delay / 2)

            # Apply action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Check if action was valid and print result
            action_valid = info.get("action_valid", True)
            print(f"Action valid: {action_valid}, Reward: {reward:.2f}")
            print(f"Total reward: {total_reward:.2f}")

            # Update episode score from info
            if "score" in info:
                episode_score = info["score"]

            # Turn off thinking visualization if renderer is available
            if hasattr(env, "renderer") and env.renderer:
                env.renderer.set_agent_thinking(False)
            elif (
                hasattr(env, "unwrapped")
                and hasattr(env.unwrapped, "renderer")
                and env.unwrapped.renderer
            ):
                env.unwrapped.renderer.set_agent_thinking(False)

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
