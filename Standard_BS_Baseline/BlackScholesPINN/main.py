import argparse
import json
import torch
from model import PINN
from data import generate_synthetic_data
from train import train
from black_scholes import BlackScholesPINN

def main(config_path):
    # 1. Load config from JSON file
    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. Generate synthetic training data
    bs = BlackScholesPINN(config)
    bs.train()
    bs.export()
    print(f"\n✅ Model saved to {config.get('model_path', 'model.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PINN on the Black-Scholes equation")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration file (default: config.json)"
    )
    args = parser.parse_args()
    main(args.config)
