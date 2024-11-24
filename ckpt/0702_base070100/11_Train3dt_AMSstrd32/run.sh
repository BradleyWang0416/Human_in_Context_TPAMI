#!/bin/bash

# 设置变量
ROOT_DIR="/data/home/SiC"   # For FLS
# ROOT_DIR="/home/wxs/Skeleton-in-Context-tpami"  # For Bradley

EXP_DIR="ckpt/0702_base070100/11_Train3dt_AMSstrd32"

DIR="${ROOT_DIR}/${EXP_DIR}"

# 创建多级目录
# mkdir -p "$DIR"

# 运行 Python 脚本，并将输出重定向到日志文件
nohup python -u ${ROOT_DIR}/train_DDP.py \
    --config "$DIR/config.yaml" \
    -c "$DIR" \
    -vertex_x1000 \
    -fully_connected_graph \
    -data_efficient \
    -gpu 0,1,2,3,4,5,6,7 \
    -bs 64 \
    > "$DIR/run_$(date +%Y%m%d).log" 2>&1 &
# bs: 320(DP)