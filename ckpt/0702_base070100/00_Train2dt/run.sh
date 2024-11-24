#!/bin/bash

# 设置变量
EXP_DIR="ckpt/0702_base070100/00_Train2dt"

# 运行 Python 脚本，并将输出重定向到日志文件
nohup python -u train_DDP.py \
    --config "$EXP_DIR/config.yaml" \
    -c "$EXP_DIR" \
    -vertex_x1000 \
    -fully_connected_graph \
    -data_efficient \
    -gpu 4,5,6,7 \
    -bs 92 \
    > "$EXP_DIR/run_$(date +%Y%m%d).log" 2>&1 &
# bs: 320(DP)