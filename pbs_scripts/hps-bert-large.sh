#!/usr/bin/bash

#PBS -q umagpu
#PBS -N bert-large-hps
#PBS -l nodes=1:ppn=16
#PBS -e pbs_output/bert-large-hps-errors
#PBS -o pbs_output/bert-large-hps-output

unset CUDA_VISIBLE_DEVICES

cd $PBS_O_WORKDIR

source venv/bin/activate

export HF_DATASETS_CACHE=venv/cache
export HF_HOME=venv/cache

MODEL_NAME="neuralmind/bert-large-portuguese-cased" 
DATASET_NAME="carolina-c4ai/carol-domain-sents"
NUM_LABELS=5
MAX_LENGTH=256
BATCH_SIZE=32
NUM_EPOCHS=3
NUM_TRIALS=100

torchrun --standalone --nproc_per_node=1 hyperparameter-search.py \
	"$MODEL_NAME"\
	"$DATASET_NAME" \
	"$NUM_LABELS" \
	"$MAX_LENGTH" \
	"$BATCH_SIZE" \
	"$NUM_EPOCHS" \
	"$NUM_TRIALS" \
	--disable_compile
