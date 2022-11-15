#!/bin/bash

NAME=$1
EXP=$2
CONFIG=$3
DATA_DIR=/esat/topaz/gkouros/datasets/nerf/$1

export PATH="/usr/local/cuda-11/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multinerf

DIR=/users/visics/gkouros/projects/nerf-repos/refnerf-pytorch/
cd ${DIR}

TF_FORCE_GPU_ALLOW_GROWTH='true' python3 eval.py \
  --gin_configs=$CONFIG \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DIR}/logs/$1/$2'" \
  --logtostderr

conda deactivate
