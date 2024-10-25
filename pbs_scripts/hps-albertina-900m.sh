#!/usr/bin/bash

#PBS -q duasgpus
#PBS -N albertina-900m-hps
#PBS -l nodes=1:ppn=32
#PBS -e pbs_output/albertina-900m-hps-errors
#PBS -o pbs_output/albertina-900m-hps-output
#PBS -m abe

unset CUDA_VISIBLE_DEVICES

cd $PBS_O_WORKDIR

source venv/bin/activate

export HF_DATASETS_CACHE=venv/cache
export HF_HOME=venv/cache

MODEL_NAME="PORTULAN/albertina-900m-portuguese-ptbr-encoder"
DATASET_NAME="mmcarpi/caroldb-sentences"
NUM_LABELS=5
MAX_LENGTH=256
# BATCH_SIZE
NUM_EPOCHS=3
NUM_TRIALS=100

BATCH_SIZES=(32 16 8 4 2)

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    echo "Trying to train with batch size: $BATCH_SIZE"
	        
    if torchrun --standalone --nproc_per_node=2 hyperparameter-search.py "$MODEL_NAME" "$DATASET_NAME" "$NUM_LABELS" "$MAX_LENGTH" "$BATCH_SIZE" "$NUM_EPOCHS" "$NUM_TRIALS" --iters_to_accumulate=$(( ${BATCH_SIZES[0]} / $BATCH_SIZE )); then
	    echo "Training succeeded with batch size: $BATCH_SIZE"
	    exit 0  # Exit if training is successful
    else
	    echo "Training failed with batch size: $BATCH_SIZE"
    fi
done

echo "All attempts failed. Exiting."
exit 1
