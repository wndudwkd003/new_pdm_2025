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
    return parser.parse_args()


def build_config():
    config = Config()
    if config.model.model_size != ModelSize.NONE:
        config.params = MODEL_SIZE_MAP[config.model.model_size]
    return config

def main(config: Config):
    trainer = Trainer(config)
    args = parse_args()
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
    #todo: yaml로 파싱해서 불러오는거 해야함
    config = build_config()
    # 시드 고정 확실히 됨
    set_seeds(config.train.seed)
    read_token_and_os_export()
    main(config)




