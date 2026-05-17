"""
Evaluate a trained kick-boxing agent with visualization.

Usage:
    python scripts/evaluate.py --model checkpoints/rl/best/best_model.zip --render
    python scripts/evaluate.py --model checkpoints/rl/kickboxing_ppo_final.zip --episodes 50
"""
import argparse
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from envs.kickboxing_env import KickBoxingEnv
from utils.config import load_config


def evaluate(args):
    config = load_config(args.config)

    render_mode = "human" if args.render else None
    env = KickBoxingEnv(config=config, render_mode=render_mode)

    model = PPO.load(args.model)

    all_rewards = []
    all_lengths = []
    all_strikes = []
    all_strike_details = {"head": 0, "torso": 0, "legs": 0}

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0
        ep_strikes = 0
        step = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            ri = info.get("reward_info", {})
            ep_strikes += ri.get("strike", 0)

            if args.render:
                env.render()
                time.sleep(0.02)

            done = terminated or truncated

        all_rewards.append(total_reward)
        all_lengths.append(step)
        all_strikes.append(ep_strikes)

        print(
            f"Episode {ep + 1}/{args.episodes}: "
            f"reward={total_reward:.2f}, "
            f"length={step}, "
            f"strikes={ep_strikes:.2f}, "
            f"terminated={'fall' if terminated else 'timeout'}"
        )

    env.close()

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Episodes:       {args.episodes}")
    print(f"Mean Reward:    {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
    print(f"Mean Length:    {np.mean(all_lengths):.1f} +/- {np.std(all_lengths):.1f}")
    print(f"Mean Strikes:   {np.mean(all_strikes):.2f}")
    print(f"Max Reward:     {np.max(all_rewards):.2f}")
    print(f"Min Reward:     {np.min(all_rewards):.2f}")
    print(f"Survival Rate:  {sum(1 for l in all_lengths if l >= 1000) / args.episodes * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate kick-boxing agent")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Enable visualization")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
