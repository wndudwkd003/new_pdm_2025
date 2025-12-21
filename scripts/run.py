# scripts/run.py

import argparse
import os
import yaml
from src.configs.configs import Config
from src.trainer.trainer import Trainer
from src.params.data_model import Split
from src.utils.seed_util import set_seeds
from src.params.model_map import MODEL_SIZE_MAP
from src.params.data_model import ModelSize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--load_dir", type=str, default=None)


    return parser.parse_args()


def build_config(args):
    config = Config()

    if config.model.model_size != ModelSize.NONE:
        config.params = MODEL_SIZE_MAP[config.model.model_size]

    # seed 설정
    if args.seed is not None:
        config.train.seed = args.seed

    # ★ load_dir 설정 (test 모드)
    if args.load_dir is not None:
        config.model.save_work_dir = args.load_dir

    return config


def main(config: Config, args):
    trainer = Trainer(config)
    results_dir = trainer.run(Split[args.mode.upper()])
    print(f"[{args.mode}] Results are saved in: {results_dir}")


def read_token_and_os_export():
    with open("src/configs/token.yaml", 'r') as f:
        token_config = yaml.safe_load(f)
    hf_token = token_config.get("hf_token", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("[Token] HF_TOKEN environment variable set.")
    else:
        print("[Token] No HF_TOKEN found in token.yaml.")

if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)
    set_seeds(config.train.seed)
    read_token_and_os_export()
    main(config, args)




