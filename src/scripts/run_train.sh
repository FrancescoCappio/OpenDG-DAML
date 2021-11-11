#!/usr/bin/env bash
# OfficeHome

cd ..

gpu=0

ipe=101

source=ACP
target=R

log_name=daml-${source}-${target}

python daml.py \
    --root dataset/OfficeHome/ \
    -d OfficeHome \
    -s ${source} \
    -t ${target} \
    --gpu ${gpu} \
    --iters_per_epoch ${ipe} \
    --savename ${log_name}
