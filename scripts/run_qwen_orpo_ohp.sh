#!/bin/bash

# Qwen-ORPO series are trained on 8 * H100s

# 0.5B

accelerate launch --config_file ./src/accelerate/fsdp.yaml main.py \
    --lr 5e-6 \
    --torch_compile False \
    --alpha 0.05 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_steps 100 \
    --model_name Qwen/Qwen1.5-0.5B \
    --data_name argilla/OpenHermesPreferences \
    --num_train_epochs 3 \
    --optim adamw_torch \
    --gradient_accumulation_steps 1 \
    --prompt_max_length 1792 \
    --response_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --num_proc 32 \
    --flash_attention_2 \
    --push_to_hub true \
    --save_dir data/Qwen1.5-0.5B-ohp-v0.1 \
    --hub_model_id orpo-explorers/Qwen1.5-0.5B-ohp-v0.1

    # 7B

    accelerate launch --config_file ./src/accelerate/fsdp.yaml main.py \
    --lr 5e-6 \
    --torch_compile False \
    --alpha 0.05 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_steps 100 \
    --model_name Qwen/Qwen1.5-7B \
    --data_name orpo-explorers/OpenHermesPreferences-10k \
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
    --save_dir data/Qwen1.5-7B-ohp-10k-v0.1 \
    --hub_model_id orpo-explorers/Qwen1.5-7B-ohp-10k-v0.1