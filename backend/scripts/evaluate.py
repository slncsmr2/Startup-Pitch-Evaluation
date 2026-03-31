from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.trainer import evaluate_from_config


DEFAULT_CONFIG = Path("models/config/training_cpu.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate phase-6 multimodal neural scoring model.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to training/evaluation config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path. If omitted, default checkpoint from config is used.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = args.checkpoint if args.checkpoint else None
    result = evaluate_from_config(config_path=args.config, checkpoint_path=checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
