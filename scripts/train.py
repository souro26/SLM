"""
scripts/train.py

Launch script for training the SLM.
Parses the configuration and runs the Trainer.

Usage:
    python -m scripts.train --config configs/train_pilot.yaml
"""

import argparse
import logging
import sys

from model.config import ModelConfig
from model.transformer import TransformerModel
from train.config import TrainConfig
from train.trainer import Trainer

# Setup basic console logging for the startup phase
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SLM model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the training YAML config file"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model.yaml",
        help="Path to the model YAML config file",
    )
    args = parser.parse_args()

    logger.info("Loading training config from %s", args.config)
    train_cfg = TrainConfig.from_yaml(args.config)

    logger.info("Loading model config from %s", args.model_config)
    model_cfg = ModelConfig.from_yaml(args.model_config)

    logger.info("Initializing model...")
    model = TransformerModel(model_cfg)

    logger.info("Initializing trainer...")
    trainer = Trainer(train_cfg, model)

    trainer.run()


if __name__ == "__main__":
    main()
