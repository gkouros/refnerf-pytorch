#!/bin/bash

# cd to the dir of the script so that you can execute it from anywhere
DIR=$( realpath -e -- $( dirname -- ${BASH_SOURCE[0]}))
cd $DIR
echo $DIR

# ARGS="Model.ray_shape=\'line\'"
# condor_send --jobname "train_ref_pytorch_shiny_car_$ARGS" --conda-env 'refnerf' --gpumem 65 --mem 32 --gpus 1 --cpus 2 --timeout 2.99 --nice 1 \
#   -c "bash train.sh ref_shiny/car EXP configs/blender_refnerf.gin $ARGS"

# EXP="14256"
# condor_send --jobname "train_ref_pytorch_shiny_car_$ARGS" --conda-env 'refnerf' --gpumem 12 --mem 32 --gpus 1 --cpus 2 --timeout 1.00 --nice 1 \
#   -c "bash eval.sh ref_shiny/car $EXP configs/blender_refnerf.gin $ARGS"

# EXP="14260"
# condor_send --jobname "train_ref_pytorch_shiny_car_$ARGS" --conda-env 'refnerf' --gpumem 16 --mem 32 --gpus 1 --cpus 2 --timeout 1.00 --nice 1 \
#   -c "bash eval.sh ref_shiny/car $EXP configs/blender_refnerf.gin $ARGS"
