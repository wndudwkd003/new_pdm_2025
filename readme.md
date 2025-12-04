
### auto_run.sh 실행 방법

- auto_run.sh 안에 json 넣어야 함

```bash
./runs/auto_run.sh {gpu} {train/test}

```

train 예시
./runs/auto_run.sh 0 train

test 예시
./runs/auto_run.sh 0 test




종합 집계 하는 방법

./runs/agg_seeds.sh
