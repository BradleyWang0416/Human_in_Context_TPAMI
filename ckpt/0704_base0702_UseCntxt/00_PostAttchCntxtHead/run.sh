#!/bin/bash

# 设置变量
EXP_DIR="ckpt/0704_base0702_UseCntxt/00_PostAttchCntxtHead"

# 运行 Python 脚本，并将输出重定向到日志文件
nohup python -u train_DDP.py \
    --config "$EXP_DIR/config.yaml" \
    -c "$EXP_DIR" \
    -vertex_x1000 \
    -fully_connected_graph \
    -data_efficient \
    -use_context post_attach_context_head \
    -gpu 4,5,6,7 \
    -bs 92 \
    > "$EXP_DIR/run_$(date +%Y%m%d).log" 2>&1 &
# bs: 320(DP)