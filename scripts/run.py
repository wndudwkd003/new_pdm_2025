# scripts/run.py

import argparse

from src.configs.configs import Config
from src.trainer.trainer import Trainer
from src.utils.seed_util import set_seeds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    return parser.parse_args()


def build_config():
    config = Config()
    return config

def main(config: Config):

    trainer = Trainer(config)

    args = parse_args()

    results_dir = trainer.run(args.mode)

    print(f"[{args.mode}] Results are saved in: {results_dir}")


if __name__ == "__main__":
    #todo: yaml로 파싱해서 불러오는거 해야함
    config = build_config()

    # 시드 고정 확실히 됨
    set_seeds(config.train.seed)

    main(config)




