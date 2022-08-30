#!/bin/bash

NAME=$1
EXP=$2
DATA_DIR=/esat/topaz/gkouros/datasets/nerf/$1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multinerf

export PATH="/usr/local/cuda-11/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11/lib64:/usr/local/cuda/lib64:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH"

DIR=/users/visics/gkouros/projects/nerf-repos/multinerf/
cd ${DIR}

python3 train.py \
  --gin_configs=configs/llff_refnerf.gin \
  --gin_bindings="Config.data_dir = '${DIR}/data/$1'" \
  --gin_bindings="Config.checkpoint_dir = '${DIR}/logs/$1'" \
  --logtostderr

conda deactivate