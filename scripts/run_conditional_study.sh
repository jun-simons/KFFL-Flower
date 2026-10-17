#!/usr/bin/env bash
# Conditional vs unconditional KFFL, plus the equalised-odds baseline, for one
# dataset:  bash scripts/run_conditional_study.sh adult|compas
#
# Runs the simulations in-process (scripts/run_inprocess.py).  Every sweep appends each finished run to its CSV and skips rows
# already there, and the baseline caches its FedAvg models, so re-running this
# script after an interruption picks up where it stopped.
set +u
DATASET=${1:?usage: run_conditional_study.sh adult|compas}
SEEDS=0,1,2,3,4
JOBS=${JOBS:-10}
# Algorithm 1 with one consistent alpha = 0.1, run long enough to converge.
SCHED='num-server-rounds=300 fairness-step-size=0.1 prox-mu=5.0'

case "$DATASET" in
  adult)  SENS=();                              NONE=0,3,10,30,100,300; COND=3,10,30,100,300 ;;
  compas) SENS=(--sensitive african-american);  NONE=0,5,7,10,15,20,30; COND=5,10,20,30,50,100 ;;
  *) echo "unknown dataset $DATASET"; exit 1 ;;
esac

sweep() {  # sweep <name> <partition> <lambdas> <extra run-config>
  python scripts/sweep.py --dataset "$DATASET" "${SENS[@]}" --partition "$2" \
    --seeds "$SEEDS" --jobs "$JOBS" --inprocess --lambdas "$3" --extra "$SCHED $4" \
    --out "results/cond_${DATASET}_$1_$2.csv" >> "results/logs/cond_${DATASET}_$1_$2.log" 2>&1
}

mkdir -p results/logs
for PART in non-iid iid; do
  python scripts/eo_baseline.py --dataset "$DATASET" --partition "$PART" --seeds "$SEEDS" \
    >> "results/logs/eo_baseline_${DATASET}_$PART.log" 2>&1
  sweep none "$PART" "$NONE" ''
  sweep label-prevalence "$PART" "$COND" 'khsic-conditioning="label" khsic-stratum-weights="prevalence"'
  sweep label-uniform "$PART" "$COND" 'khsic-conditioning="label" khsic-stratum-weights="uniform"'
done
echo "STUDY $DATASET DONE"
