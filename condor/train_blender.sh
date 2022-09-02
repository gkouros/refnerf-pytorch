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

TF_FORCE_GPU_ALLOW_GROWTH='true' python3 train.py \
  --gin_configs=configs/blender_refnerf.gin \
  --gin_bindings="Config.data_dir = '${DIR}/data/$1'" \
  --gin_bindings="Config.checkpoint_dir = '${DIR}/logs/$1/$2'" \
  --logtostderr \
  & \
  TF_FORCE_GPU_ALLOW_GROWTH='true' python3 render.py \
    --gin_configs=configs/blender_refnerf.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.checkpoint_dir = '${DIR}/logs/$1/$2'" \
    --gin_bindings="Config.render_dir = '${DIR}/logs/$1/$2/render/'" \
    --gin_bindings="Config.render_path = True" \
    --gin_bindings="Config.render_path_frames = 480" \
    --gin_bindings="Config.render_video_fps = 60" \
    --logtostderr

conda deactivate
