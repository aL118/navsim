#!/bin/bash

#SBATCH --job-name=navhard_eval
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/navsim/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/navsim/my_dump/%x.out.%j

## Scale ntasks with gpus
#SBATCH --mem=120gb                                               # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:rtxa5000:8
#SBATCH --ntasks=32

## GAMMA training config
#SBATCH --time=58:00:00     
#SBATCH --qos=huge-long                                    
#SBATCH --account=gamma
#SBATCH --partition=gamma

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

# Use navhard_two_stage for EPDMS evaluation
TRAIN_TEST_SPLIT=navhard_two_stage
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/navsim/models/transfuser/transfuser_seed_0.ckpt
CHECKPOINT=/fs/nexus-scratch/aliu1237/carla_garage_clone/logs/BASELINE/model_0029.pth
CACHE_PATH=/fs/nexus-projects/sim2real/aliu/navsim/metric_cache_navhard

# For navhard_two_stage, both original and synthetic sensor blobs are in the same directory
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
  train_test_split=$TRAIN_TEST_SPLIT \
  agent=carla_garage_agent \
  worker=single_machine_thread_pool \
  agent.checkpoint_path=$CHECKPOINT \
  experiment_name=carla_garage_navhard \
  metric_cache_path=$CACHE_PATH \
  original_sensor_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs \
  synthetic_sensor_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs \
  synthetic_scenes_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

# Submit with: sbatch scripts/evaluation/navhard_sbatch.sh
