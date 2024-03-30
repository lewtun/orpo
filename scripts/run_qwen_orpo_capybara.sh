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
    --data_name argilla/distilabel-capybara-dpo-7k-binarized \
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
    --save_dir data/Qwen1.5-0.5B-capybara-dpo-7k-v0.1 \
    --hub_model_id orpo-explorers/Qwen1.5-0.5B-capybara-dpo-7k-v0.1
    

# 72B - can't fit on 1 node :(

accelerate launch --config_file ./src/accelerate/fsdp.yaml main.py \
    --lr 5e-6 \
    --torch_compile False \
    --alpha 0.05 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_steps 100 \
    --model_name Qwen/Qwen1.5-72B \
    --data_name argilla/distilabel-capybara-dpo-7k-binarized \
    --num_train_epochs 3 \
    --optim adamw_torch \
    --gradient_accumulation_steps 4 \
    --prompt_max_length 1792 \
    --response_max_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --num_proc 32 \
    --flash_attention_2 \
    --push_to_hub true \
    --save_dir data/Qwen1.5-72B-capybara-dpo-7k-v0.1 \
    --hub_model_id orpo-explorers/Qwen1.5-72B-capybara-dpo-7k-v0.1 \
    --max_steps 1