#!/bin/bash

NAME=$1
EXP=$2
CONFIG=$3
DATA_DIR=/esat/topaz/gkouros/datasets/nerf/$NAME

export PATH="/usr/local/cuda-12/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES/CUDA/}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate refnerf

DIR=/users/visics/gkouros/projects/nerf-repos/refnerf-pytorch/
cd ${DIR}

DEG_VIEW=5
RENDER_CHUNK_SIZE=8192

if [[ "$CONFIG" == *"llff"* ]]; then
  RENDER_PATH=True
else
  RENDER_PATH=False
fi

python3 render.py \
  --gin_configs="${DIR}/logs/$NAME/$EXP/config.gin" \
  --gin_bindings="Config.data_dir = '${DIR}/data/$NAME'" \
  --gin_bindings="Config.checkpoint_dir = '${DIR}/logs/$NAME/$EXP'" \
  --gin_bindings="Config.render_dir = '${DIR}/logs/$NAME/$EXP/render/'" \
  --gin_bindings="Config.render_path = $RENDER_PATH" \
  --gin_bindings="Config.render_path_frames = 480" \
  --gin_bindings="Config.render_video_fps = 60" \
  --gin_bindings="Config.render_chunk_size = $RENDER_CHUNK_SIZE" \
  --gin_bindings="NerfMLP.deg_view = $DEG_VIEW" \
  --logtostderr

conda deactivate