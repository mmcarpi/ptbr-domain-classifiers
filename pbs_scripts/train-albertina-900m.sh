#!/usr/bin/bash

#PBS -q duasgpus
#PBS -N albertina-900m-train
#PBS -l nodes=1:ppn=32
#PBS -e pbs_output/albertina-900m-train-errors
#PBS -o pbs_output/albertina-900m-train-output
#PBS -m abe

unset CUDA_VISIBLE_DEVICES

cd $PBS_O_WORKDIR

source venv/bin/activate

export HF_DATASETS_CACHE=venv/cache
export HF_HOME=venv/cache

CONFIG="Config/albertina-900m.json"
NUM_EPOCHS=5
EVALS_PER_EPOCH=2
LOG_FREQUENCY=10
NUM_WORKERS=4

torchrun --standalone --nproc_per_node=2 dist.py \
	"$CONFIG" \
	"$NUM_EPOCHS" \
	"$EVALS_PER_EPOCH" \
	--log_frequency="$LOG_FREQUENCY" \
	--num_workers="$NUM_WORKERS"
