#!/bin/bash

# 设置变量
DIR="/home/wxs/Skeleton-in-Context-tpami/ckpt/0802_CombineTasks_TrainSimul/10_Train3dt"

# 创建多级目录
mkdir -p "$DIR"

# 运行 Python 脚本，并将输出重定向到日志文件
nohup python -u /home/wxs/Skeleton-in-Context-tpami/train.py \
    --config "$DIR/config.yaml" \
    -c "$DIR" \
    -vertex_x1000 \
    -fully_connected_graph \
    -train_simultaneously \
    -gpu 0 \
    -bs 92 \
    > "$DIR/run_$(date +%Y%m%d).log" 2>&1 &
# bs: 320(DP)