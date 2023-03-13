#!/bin/bash

NAME=$1
EXP=$2
CONFIG=$3
# format the arguments to gin bindings
ARGS=${@:4} # all subsequent args are assumed args for the python script
ARGS=($ARGS)  # split comma separated string
ARGS_STR=''

for (( i=0; i<${#ARGS[@]}; ++i ));
do
  ARGS_STR="$ARGS_STR --gin_bindings=${ARGS[$i]}"
done
echo ARGS="$ARGS_STR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate refnerf

export PATH="/usr/local/cuda-12/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64:/usr/local/cuda/lib64:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES/CUDA/}

DATA_DIR=/esat/topaz/gkouros/datasets/nerf/$NAME
DIR=/users/visics/gkouros/projects/nerf-repos/refnerf-pytorch/
cd ${DIR}

DEG_VIEW=5
BATCH_SIZE=1024
RENDER_CHUNK_SIZE=8192
MAX_STEPS=250000

if [[ "$CONFIG" == *"llff"* ]]; then
  RENDER_PATH=True
else
  RENDER_PATH=False
fi

# If job gets evicted reload generated config file not original that might have been modified
if [ -f "${DIR}/logs/$NAME/$EXP/config.gin" ]; then
  CONFIG_PATH="${DIR}/logs/$NAME/$EXP/config.gin"
else
  CONFIG_PATH="$CONFIG"
fi

python3 train.py \
  --gin_configs="$CONFIG_PATH" \
  --gin_bindings="Config.max_steps = $MAX_STEPS" \
  --gin_bindings="Config.data_dir = '${DIR}/data/$NAME'" \
  --gin_bindings="Config.checkpoint_dir = '${DIR}/logs/$NAME/$EXP'" \
  --gin_bindings="Config.batch_size = $BATCH_SIZE" \
  --gin_bindings="Config.render_chunk_size = $RENDER_CHUNK_SIZE" \
  --gin_bindings="NerfMLP.deg_view = $DEG_VIEW" \
  $ARGS_STR \
  && \
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
  && \
  python3 eval.py \
  --gin_configs="${DIR}/logs/$NAME/$EXP/config.gin" \
  --gin_bindings="Config.data_dir = '${DIR}/data/$NAME'" \
  --gin_bindings="Config.checkpoint_dir = '${DIR}/logs/$NAME/$EXP'" \
  --gin_bindings="Config.render_chunk_size = $RENDER_CHUNK_SIZE" \
  --gin_bindings="NerfMLP.deg_view = $DEG_VIEW"

conda deactivate
