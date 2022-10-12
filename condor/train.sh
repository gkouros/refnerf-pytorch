#!/bin/bash

NAME=$1
EXP=$2
CONFIG=$3
DATA_DIR=/esat/topaz/gkouros/datasets/nerf/$1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multinerf

export PATH="/usr/local/cuda-11/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11/lib64:/usr/local/cuda/lib64:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH"

DIR=/users/visics/gkouros/projects/nerf-repos/Ref-NeRF-plusplus/
cd ${DIR}

ENABLE_PRED_ROUGHNESS=True
DEG_VIEW=5
BATCH_SIZE=1024
RENDER_CHUNK_SIZE=1024

XLA_PYTHON_CLIENT_ALLOCATOR=platform TF_FORCE_GPU_ALLOW_GROWTH='true' python3 train.py \
  --gin_configs="$CONFIG" \
  --gin_bindings="Config.data_dir = '${DIR}/data/$1'" \
  --gin_bindings="Config.checkpoint_dir = '${DIR}/logs/$1/$2'" \
  --gin_bindings="Config.batch_size = $BATCH_SIZE" \
  --gin_bindings="Config.render_chunk_size = $RENDER_CHUNK_SIZE" \
  --gin_bindings="NerfMLP.deg_view = $DEG_VIEW" \
  --gin_bindings="NerfMLP.enable_pred_roughness = $ENABLE_PRED_ROUGHNESS" \
  --logtostderr \
  && \
  python3 render.py \
    --gin_configs="${DIR}/logs/$1/$2/config.gin" \
    --gin_bindings="Config.data_dir = '${DIR}/data/$1'" \
    --gin_bindings="Config.checkpoint_dir = '${DIR}/logs/$1/$2'" \
    --gin_bindings="Config.render_dir = '${DIR}/logs/$1/$2/render/'" \
    --gin_bindings="Config.render_path = False" \
    --gin_bindings="Config.render_path_frames = 480" \
    --gin_bindings="Config.render_video_fps = 60" \
    --gin_bindings="Config.batch_size = $BATCH_SIZE" \
    --gin_bindings="Config.render_chunk_size = $RENDER_CHUNK_SIZE" \
    --gin_bindings="NerfMLP.deg_view = $DEG_VIEW" \
    --gin_bindings="NerfMLP.enable_pred_roughness = $ENABLE_PRED_ROUGHNESS" \
    --logtostderr \
  && \
  python3 eval.py \
  --gin_configs="${DIR}/logs/$1/$2/config.gin" \
  --gin_bindings="Config.data_dir = '${DIR}/data/$1" \
  --gin_bindings="Config.checkpoint_dir = '${DIR}/logs/$1/$2'" \
  --gin_bindings="Config.batch_size = $BATCH_SIZE" \
  --gin_bindings="Config.render_chunk_size = $RENDER_CHUNK_SIZE" \
  --gin_bindings="NerfMLP.deg_view = $DEG_VIEW" \
  --gin_bindings="NerfMLP.enable_pred_roughness = $ENABLE_PRED_ROUGHNESS" \
  --logtostderr

conda deactivate
