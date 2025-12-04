import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


# ----------------------------------------------------------
# Argument Parser
# ----------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True)

    # 여러 GPU를 지원: --gpus 0 1 2
    parser.add_argument("--gpus", type=int, nargs="+", required=True)

    # auto_run.sh 에서 seeds 넘김 (train 모드에서 사용)
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
# TRAIN 모드 실행 (여러 GPU 병렬)
# ----------------------------------------------------------
def run_train_mode(args):
    gpus = args.gpus
    seeds = args.seeds
    mode = "train"

    if len(gpus) == 0:
        raise ValueError("적어도 하나의 GPU를 --gpus 로 전달해야 합니다.")

    if len(seeds) == 0:
        raise ValueError("적어도 하나의 seed 를 --seeds 로 전달해야 합니다.")

    # GPU별로 seed를 round-robin으로 분배
    assignments = {gpu: [] for gpu in gpus}
    for idx, seed in enumerate(seeds):
        gpu = gpus[idx % len(gpus)]
        assignments[gpu].append(seed)

    # GPU 하나가 담당하는 시드들을 순차적으로 실행하는 worker
    def worker(gpu_id: int, worker_seeds: list[int]):
        worker_runs: list[dict] = []

        for seed in worker_seeds:
            cmd = ["./runs/run.sh", str(gpu_id), mode, str(seed)]
            out = run_sh(cmd)

            marker = "Results are saved in:"
            idx = out.find(marker)

            if idx != -1:
                results_dir_raw = out[idx + len(marker):].strip()
                root_dir = str(Path(results_dir_raw).parent)
                worker_runs.append(
                    {
                        "seed": seed,
                        "dir": root_dir,
                        "gpu": gpu_id,
                    }
                )
            else:
                print(f"[auto_run][WARN] results_dir not found for seed {seed} (gpu {gpu_id})")

        return worker_runs

    runs: list[dict] = []

    # GPU 개수만큼 worker를 병렬로 실행 (GPU마다 자기 할당된 시드들을 순차 처리)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = []

        for gpu in gpus:
            worker_seeds = assignments[gpu]
            if len(worker_seeds) > 0:
                f = executor.submit(worker, gpu, worker_seeds)
                futures.append(f)

        for f in futures:
            worker_runs = f.result()
            for r in worker_runs:
                runs.append(r)

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


# ----------------------------------------------------------
# TEST 모드 실행
# ----------------------------------------------------------
def run_test_mode(args):
    path = Path(args.json_path)

    with open(path, "r") as f:
        data = json.load(f)

    runs = data["runs"]
    gpus = args.gpus

    if len(gpus) == 0:
        raise ValueError("적어도 하나의 GPU를 --gpus 로 전달해야 합니다.")

    if len(runs) == 0:
        print("[auto_run][TEST] runs 가 비어 있습니다. 종료합니다.")
        return

    # GPU별로 run을 round-robin으로 분배
    assignments = {gpu: [] for gpu in gpus}
    for idx, item in enumerate(runs):
        gpu = gpus[idx % len(gpus)]
        assignments[gpu].append(item)

    def worker(gpu_id: int, worker_runs: list[dict]):
        for item in worker_runs:
            seed = item["seed"]
            dir_path = item["dir"]

            cmd = ["./runs/run.sh", str(gpu_id), "test", str(seed), dir_path]
            print(f"[auto_run][TEST] GPU {gpu_id} ← seed {seed}, dir {dir_path}")
            subprocess.call(cmd)

    # GPU 개수만큼 worker를 병렬로 실행 (GPU마다 자기 할당된 run들을 순차 처리)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = []

        for gpu in gpus:
            worker_runs = assignments[gpu]
            if len(worker_runs) > 0:
                f = executor.submit(worker, gpu, worker_runs)
                futures.append(f)

        # 여기서 모두 끝날 때까지 기다림
        for f in futures:
            f.result()



def main():
    args = parse_args()

    if args.mode == "train":
        runs = run_train_mode(args)
        save_train_summary(runs)

    elif args.mode == "test":
        run_test_mode(args)


if __name__ == "__main__":
    main()
