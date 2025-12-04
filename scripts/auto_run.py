import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime


# ----------------------------------------------------------
# Argument Parser
# ----------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True)
    parser.add_argument("--gpu", type=int, required=True)

    # auto_run.sh 에서 seeds 넘김
    parser.add_argument("--seeds", type=int, nargs="+", default=[])

    # test 모드에서 summary json 경로
    parser.add_argument("--json_path", type=str, default=None)

    return parser.parse_args()


# ----------------------------------------------------------
# run.sh 실행 후 전체 output 반환
# ----------------------------------------------------------
def run_sh(cmd):
    out = subprocess.check_output(cmd, text=True)
    print(out)
    return out


# ----------------------------------------------------------
# TRAIN 모드 실행
# ----------------------------------------------------------
def run_train_mode(args):
    gpu = args.gpu
    seeds = args.seeds
    mode = "train"

    runs = []

    for seed in seeds:
        cmd = ["./runs/run.sh", str(gpu), mode, str(seed)]
        out = run_sh(cmd)

        marker = "Results are saved in:"
        idx = out.find(marker)
        if idx == -1:
            print(f"[auto_run][WARN] results_dir not found for seed {seed}")
            continue

        # raw 경로: .../history
        results_dir_raw = out[idx + len(marker):].strip()

        # history 제거 → 상위 폴더 기록
        root_dir = str(Path(results_dir_raw).parent)

        runs.append({
            "seed": seed,
            "dir": root_dir,
        })

    return runs


# ----------------------------------------------------------
# TRAIN 요약 JSON 저장
# ----------------------------------------------------------
def save_train_summary(runs: list[dict]):

    sample = runs[0]
    root_dir = Path(sample["dir"])
    name = root_dir.name

    parts = name.split("_")

    # timestamp
    timestamp = parts[0] + "_" + parts[1]

    # model
    model_name = parts[2]

    # seed 숫자만 추출 (필요하면)
    seed_str = parts[3].replace("seed", "")

    # scenario: seed 다음 모든 파트
    scenario = "_".join(parts[4:])


    # 저장 파일명
    json_name = f"{timestamp}_{model_name}_{scenario}_auto.json"

    save_dir = Path("outputs_auto_run")
    save_dir.mkdir(exist_ok=True)

    save_path = save_dir / json_name

    obj = {
        "timestamp": timestamp,
        "model": model_name,
        "scenario": scenario,
        "runs": runs,
    }

    with open(save_path, "w") as f:
        json.dump(obj, f, indent=4)

    print(f"[auto_run][TRAIN] Summary saved: {save_path}")


def run_test_mode(args):
    path = Path(args.json_path)

    with open(path, "r") as f:
        data = json.load(f)

    runs = data["runs"]

    for item in runs:
        seed = item["seed"]
        dir_path = item["dir"]

        cmd = ["./runs/run.sh", str(args.gpu), "test", str(seed), dir_path]
        subprocess.call(cmd)


def main():
    args = parse_args()

    if args.mode == "train":
        runs = run_train_mode(args)
        save_train_summary(runs)


    elif args.mode == "test":
        run_test_mode(args)


if __name__ == "__main__":
    main()
