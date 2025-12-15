#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=3
export DS_ACCELERATOR=musa
export LOGLEVEL="INFO"
export MUSA_EXECUTION_TIMEOUT=20000000
export MUSA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export MUSA_KERNEL_TIMEOUT=3200000
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0
export OMP_NUM_THREADS=4
export MCCL_ALGOS=1
export MCCL_BUFFSIZE=20971520
export MUSA_BLOCK_SCHEDULE_MODE=1
export MCCL_IB_GID_INDEX=3
export MCCL_NET_SHARED_BUFFERS=0
export MCCL_IB_TC=106
export MCCL_IB_QPS_PER_CONNECTION=16
export MCCL_IB_TIMEOUT=20
export MCCL_IB_RETRY_CNT=7
export MCCL_SOCKET_IFNAME=bond0
export MCCL_CROSS_NIC=0
export MUSA_LAUNCH_BLOCKING=0 
export DS_SKIP_CUDA_CHECK=1
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled
export PATH=$PATH:/usr/local/musa/bin:/usr/local/musa/mudnn/bin:/usr/local/musa/mudnn_bench/bin:/usr/local/musa/mccl_test:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/musa/lib:/usr/local/openmpi/lib
export TORCH_DIST_LOG_LEVEL=ERROR
export DEEPSPEED_LOG_LEVEL=ERROR

export HOST_GPU_NUM=8
# 当前机器ip
export LOCAL_IP=${10.202.43.46}
# 多节点机器ip，逗号隔开
export NODE_IP_LIST="${10.202.43.46}:8"
# 机器节点个数
export NODES=1
export NODE_NUM=$((${NODES} * ${HOST_GPU_NUM}))

export NCCL_DEBUG=WARN

# 给一个空目录，走else分支通过传参控制超参
model_path=./models
tokenizer_path=../models
train_data_file=am_0.9M_sample_1k.jsonl

# ds_config_file=ds_zero2_no_offload.json
ds_config_file=ds_zero3_no_offload.json
# ds_config_file=ds_zero3_offload_no_auto.json

output_path=./hf_train_output

mkdir -p ${output_path}

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
log_file=${output_path}/"log_${current_time}.txt"

echo $NODE_IP_LIST > env.txt 2>&1 &
sed "s/:/ slots=/g" env.txt | sed "s/,/\n/g" >  "hostfile"
sed "s/:.//g" env.txt | sed "s/,/\n/g" >  "pssh.hosts"
export CHIEF_IP=$LOCAL_IP

HOST_PATH=hostfile
# HOST_PATH=none

deepspeed --num_gpus 8 train.py \
    --do_train \
    --model_name_or_path ${model_path} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --train_data_file ${train_data_file} \
    --deepspeed ${ds_config_file} \
    --output_dir ${output_path} \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine_with_min_lr \
    --logging_steps 1 \
    --max_steps 200 \
    --save_steps 100 \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --warmup_ratio 0.01 \
    --save_strategy steps \
    --save_safetensors False \
    --hidden_size 6400 \
    --intermediate_size 18304 \
    --model_max_length 4096 \
    --max_seq_length 4096 \
    --moe_topk 1 \
    --num_experts 2 \
    --num_attention_heads 80 \
    --num_key_value_heads 8 \
    --num_layers 4 \
    --cla_share_factor 2 \
    --use_cla \
    --use_mixed_mlp_moe \
    --num_shared_expert 1 \
    --use_qk_norm \
    --use_pack_kv \
    --use_torch_rmsnorm \
    --bf16 | tee ${log_file}

    # --gradient_checkpointing \
    # --use_lora \
    # --lora_rank 64 \
    # --lora_alpha 128 \
    # --lora_dropout 0.1 \
