#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod  # Adjust this for your cluster
#SBATCH --output=/fsx/h4/logs/%x-%j.out # Adjust this for your cluster
#SBATCH --err=/fsx/h4/logs/%x-%j.err    # Adjust this for your cluster

set -x -e

source ~/.bashrc
conda activate orpo
echo "START TIME: $(date)"

ACCELERATOR=$1
GRAD_ACC_STEPS=$2
ALPHA=$3
LR=$4

# Training setup
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export CMD="main.py \
    --lr $LR \
    --torch_compile False \
    --alpha $ALPHA \
    --lr_scheduler_type inverse_sqrt \
    --warmup_steps 100 \
    --model_name Qwen/Qwen1.5-7B \
    --data_name orpo-explorers/OpenHermesPreferences-100k \
    --num_train_epochs 3 \
    --optim adamw_torch \
    --gradient_accumulation_steps 1 \
    --prompt_max_length 1024 \
    --response_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_proc 32 \
    --flash_attention_2 \
    --push_to_hub true \
    --save_dir data/Qwen1.5-7B-ohp-100k-v0.1 \
    --hub_model_id orpo-explorers/Qwen1.5-7B-ohp-100k-v0.1
    "

export LAUNCHER="HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file src/accelerate/$ACCELERATOR.yaml  \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --max_restarts 1 \
    --role \$(hostname -s): \
    --tee 3 \
    "

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# Specific configuration optimized for the Hugging Face Compute Cluster
# Be ye warned this may not work on other clusters!
module load cuda/12.1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD" 2>&1

echo "END TIME: $(date)"
