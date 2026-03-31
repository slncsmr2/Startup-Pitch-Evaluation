from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.trainer import train_from_config


DEFAULT_CONFIG = Path("models/config/training_cpu.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train phase-6 multimodal neural scoring model.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to training config file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_from_config(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
