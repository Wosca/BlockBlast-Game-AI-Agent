"""
Block Game – training utilities

Features
--------
* Continuing on‑policy RL with PPO or (optionally) MaskablePPO
* Robust checkpointing & TensorBoard logging

Usage (from shell)
------------------
$ python ppo_agent.py         # full pipeline

You can toggle the booleans in the __main__ section to control which
stages run (pre‑train, PPO, MaskablePPO, visualisation).

The script looks for/creates these folders:
    ./models/  – weights & checkpoints
    ./logs/    – TensorBoard summaries
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# Optional MaskablePPO
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

    maskable_ppo_available = True
except ImportError:
    maskable_ppo_available = False

from game_env import BlockGameEnv
from agent_visualizer import visualize_agent  # pragma: no cover


# ---------------------------------------------------------------------------
# util: make_env – worker fn for SubprocVecEnv
# ---------------------------------------------------------------------------


def make_env(rank: int, seed: int = 0) -> Callable[[], BlockGameEnv]:
    """Factory used by the vectorised environment."""

    def _init():
        env = BlockGameEnv()
        env = Monitor(env)
        try:
            env.reset(seed=seed + rank)  # Gymnasium API
        except TypeError:  # Older gym
            env.seed(seed + rank)
        return env

    return _init


# ---------------------------------------------------------------------------
# PPO helpers
# ---------------------------------------------------------------------------


def _standard_ppo(env, **kwargs) -> PPO:
    return PPO(
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
        ent_coef=0.05,
        **kwargs,
    )


def train_ppo(
    *,
    num_envs: int = 4,
    total_timesteps: int = 1_000_000,
    save_path: str = "./models/",
    continue_training: bool = False,
    pretrained_path: Optional[str] = None,
):
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # pick starting weights
    if continue_training and pretrained_path and os.path.exists(pretrained_path):
        print(f"[ppo] continuing from → {pretrained_path}")
        model = PPO.load(pretrained_path, env=env)
        reset_flag = False
    else:
        model = _standard_ppo(env)
        reset_flag = True

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=save_path,
        name_prefix="ppo_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_flag,
    )

    final_path = os.path.join(save_path, "final_ppo_model")
    model.save(final_path)
    print(f"[ppo] training done → {final_path}.zip")
    return model


# ---------------------------------------------------------------------------
# MaskablePPO (optional)
# ---------------------------------------------------------------------------


def _standard_masked(env, **kwargs):
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=5e-5,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.05,
        max_grad_norm=0.5,
        **kwargs,
    )


def train_masked_ppo(
    *,
    num_envs: int = 4,
    total_timesteps: int = 1_000_000,
    save_path: str = "./models/",
    continue_training: bool = False,
    pretrained_path: Optional[str] = None,
):
    if not maskable_ppo_available:
        raise RuntimeError("sb3_contrib not installed – MaskablePPO unavailable")

    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    if continue_training and pretrained_path and os.path.exists(pretrained_path):
        print(f"[masked] continuing from → {pretrained_path}")
        model = MaskablePPO.load(pretrained_path, env=env)
        reset_flag = False
    else:
        model = _standard_masked(env)
        reset_flag = True

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=save_path,
        name_prefix="masked_ppo_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_env = SubprocVecEnv([make_env(0, seed=42)])
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=100_000,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=reset_flag,
    )

    final_path = os.path.join(save_path, "final_masked_ppo_model")
    model.save(final_path)
    print(f"[masked] training done → {final_path}.zip")
    return model


# ---------------------------------------------------------------------------
# script entry‑point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)

    # ---- parameters – tweak here or use argparse ----
    num_envs = 8
    total_timesteps = 100_000_000
    continue_training = True

    train_ppo_without_masking = True
    train_ppo_with_masking = False
    visualize_ppo_without_masking = False
    visualize_ppo_with_masking = False

    # ---- RL fine‑tune ----
    if train_ppo_without_masking:
        train_ppo(
            num_envs=num_envs,
            total_timesteps=total_timesteps,
            continue_training=continue_training,
            pretrained_path="./models/final_ppo_model.zip",
        )

    if train_ppo_with_masking and maskable_ppo_available:
        train_masked_ppo(
            num_envs=num_envs,
            total_timesteps=total_timesteps,
            continue_training=continue_training,
            pretrained_path="./models/final_masked_ppo_model.zip",
        )

    # ---- visualise (optional) ----
    render_env = BlockGameEnv(render_mode="human")
    render_env = Monitor(render_env)

    if visualize_ppo_without_masking:
        agent = PPO.load("./models/final_ppo_model")
        visualize_agent(render_env, agent, episodes=10, delay=0.2, use_masks=False)

    if visualize_ppo_with_masking and maskable_ppo_available:
        agent = MaskablePPO.load("./models/final_masked_ppo_model")
        visualize_agent(render_env, agent, episodes=10, delay=0.2, use_masks=True)
