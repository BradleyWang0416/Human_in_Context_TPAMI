#!/bin/bash

# 设置变量
DIR="/home/wxs/Skeleton-in-Context-tpami/ckpt/0802_CombineTasks_TrainSimul/11_Train3dt_MoreAMSData"

# 创建多级目录
mkdir -p "$DIR"

# 运行 Python 脚本，并将输出重定向到日志文件
nohup python -u /home/wxs/Skeleton-in-Context-tpami/train_DDP.py \
    --config "$DIR/config.yaml" \
    -c "$DIR" \
    -vertex_x1000 \
    -fully_connected_graph \
    -train_simultaneously \
    -gpu 0,1,2,3,4,5,6,7 \
    -bs 64 \
    > "$DIR/run_$(date +%Y%m%d).log" 2>&1 &
# bs: 320(DP)