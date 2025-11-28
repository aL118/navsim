#!/bin/bash

# Script to evaluate transfuser on a small subset (100 scenarios)
# Run this with srun, e.g.:
#   srun --mem=64gb --gres=gpu:rtxa6000:1 --ntasks=4 --time=2:00:00 \
#        --qos=medium --account=gamma --partition=gamma \
#        scripts/evaluation/run_transfuser_metrics_subset_srun.sh
#
# Or if you already have an allocation (via salloc):
#   ./scripts/evaluation/run_transfuser_metrics_subset_srun.sh

eval "$(conda shell.bash hook)"
conda activate navsim

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Number of GPUS: $NUM_GPUS"

export HOME="/fs/nexus-projects/sim2real/aliu"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim/dataset"

# Use navmini for a smaller subset (or specify max_scenarios)
TRAIN_TEST_SPLIT=navtrain

# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/navsim/models/transfuser/transfuser_seed_0.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/navsim/models/carla_garage/pretrained_baseline_0030_0.ckpt
# CHECKPOINT=/fs/nexus-scratch/aliu1237/carla_garage_clone/logs/sim2drive_mixdata_stage2/model_0029.pth
# CHECKPOINT=/fs/nexus-scratch/aliu1237/carla_garage_clone/logs/sim2drive_stage2_19_30/model_0029.pth
CHECKPOINT=/fs/nexus-scratch/aliu1237/carla_garage_clone/logs/BASELINE/model_0029.pth
CACHE_PATH=/fs/nexus-projects/sim2real/aliu/navsim/metric_cache
# transfuser_agent
# Use run_pdm_score.py (the evaluation script without visualization)
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=carla_garage_agent \
worker=single_machine_thread_pool \
agent.checkpoint_path=$CHECKPOINT \
agent.strict_checkpoint_loading=true \
experiment_name=bash \
metric_cache_path=$CACHE_PATH \
+max_scenarios=100
