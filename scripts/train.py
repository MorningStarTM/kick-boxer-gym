"""
Main training script for kick-boxing RL agent.

Usage:
    # Phase 1: Pre-train motion VAE on mocap data
    python scripts/train.py --phase pretrain --data_dir data/mocap

    # Phase 2: Train RL agent (with pre-trained VAE)
    python scripts/train.py --phase rl --vae_path checkpoints/pretrain/motion_vae_best.pt

    # Phase 2: Train RL agent (from scratch, no pre-training)
    python scripts/train.py --phase rl

    # Both phases sequentially
    python scripts/train.py --phase all --data_dir data/mocap
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import load_config


def run_pretrain(args, config):
    from pretrain.train_motion_vae import train as pretrain

    class PretrainArgs:
        def __init__(self, data_dir, config_path, save_dir):
            self.data_dir = data_dir
            self.config = config_path
            self.save_dir = save_dir

    pretrain_args = PretrainArgs(
        data_dir=args.data_dir,
        config_path=args.config,
        save_dir=args.save_dir + "/pretrain",
    )
    pretrain(pretrain_args)


def run_rl(args, config):
    from agents.ppo_agent import KickBoxingAgent

    agent = KickBoxingAgent(
        config=config,
        pretrained_vae_path=args.vae_path,
    )

    try:
        total_steps = config.get("training", {}).get("total_timesteps", 10_000_000)
        if args.timesteps:
            total_steps = args.timesteps

        print(f"Starting RL training for {total_steps:,} timesteps...")
        agent.train(
            total_timesteps=total_steps,
            save_dir=args.save_dir + "/rl",
        )

        results = agent.evaluate(n_episodes=20)
        print("\n=== Evaluation Results ===")
        print(f"  Mean Reward:  {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
        print(f"  Mean Length:  {results['mean_length']:.1f}")
        print(f"  Mean Strikes: {results['mean_strikes']:.2f}")
    finally:
        agent.close()


def main():
    parser = argparse.ArgumentParser(description="Train kick-boxing RL agent")
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["pretrain", "rl", "all"],
        help="Training phase",
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--data_dir", type=str, default="data/mocap", help="Mocap data directory")
    parser.add_argument("--vae_path", type=str, default=None, help="Pre-trained VAE checkpoint")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.phase in ("pretrain", "all"):
        print("=" * 60)
        print("Phase 1: Pre-training Motion VAE on Mocap Data")
        print("=" * 60)
        run_pretrain(args, config)

        if args.vae_path is None:
            args.vae_path = str(Path(args.save_dir) / "pretrain" / "motion_vae_best.pt")

    if args.phase in ("rl", "all"):
        print("=" * 60)
        print("Phase 2: Training RL Agent")
        print("=" * 60)
        run_rl(args, config)


if __name__ == "__main__":
    main()
