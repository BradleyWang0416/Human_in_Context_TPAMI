#!/bin/bash

# 设置变量
# ROOT_DIR="/data/home/SiC"   # For FLS
ROOT_DIR="/home/wxs/Skeleton-in-Context-tpami"  # For Bradley

EXP_DIR="ckpt/0701/00_4gpu_DDP"

DIR="${ROOT_DIR}/${EXP_DIR}"

# 创建多级目录
# mkdir -p "$DIR"

# 运行 Python 脚本，并将输出重定向到日志文件
nohup python -u ${ROOT_DIR}/train_DDP.py \
    --config "$DIR/config.yaml" \
    -c "$DIR" \
    -vertex_x1000 \
    -fully_connected_graph \
    -gpu 4,5 \
    -bs 92 \
    > "$DIR/run_$(date +%Y%m%d).log" 2>&1 &
# bs: 320(DP)