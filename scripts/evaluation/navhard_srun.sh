#!/bin/bash

#SBATCH --job-name=test                            
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/navsim/my_dump/%x.out.%j       # indicates a file to redirect STDOUT to; %j is the jobid
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/navsim/my_dump/%x.out.%j        # indicates a file to redirect STDERR to; %j is the jobid

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

# CHECKPOINT=/fs/nexus-scratch/aliu1237/carla_garage_clone/logs/BASELINE/model_0029.pth
CHECKPOINT=/fs/nexus-scratch/aliu1237/carla_garage_clone/logs/sim2drive_mixdata_stage2/model_0029.pth
# CHECKPOINT=/fs/nexus-scratch/aliu1237/carla_garage_clone/logs/full_stage2/model_0029.pth
# CHECKPOINT2=/fs/nexus-scratch/aliu1237/carla_garage_clone/logs/sim2drive_stage2_19_30/model_0029.pth
# CHECKPOINT2=/fs/nexus-scratch/aliu1237/carla_garage_clone/logs/mixdata_stage2/model_0029.pth
CACHE_PATH=/fs/nexus-projects/sim2real/aliu/navsim/metric_cache_navhard

# For navhard_two_stage, both original and synthetic sensor blobs are in the same directory
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
  train_test_split=$TRAIN_TEST_SPLIT \
  agent=carla_garage_agent \
  +agent.config.latent=true \
  worker=single_machine_thread_pool \
  agent.checkpoint_path=$CHECKPOINT \
  experiment_name=mixdata_sim2drive \
  metric_cache_path=$CACHE_PATH \
  original_sensor_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs \
  synthetic_sensor_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs \
  synthetic_scenes_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles \
  +max_scenarios=100

# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
#   train_test_split=$TRAIN_TEST_SPLIT \
#   agent=carla_garage_agent \
#   +agent.config.latent=true \
#   worker=single_machine_thread_pool \
#   agent.checkpoint_path=$CHECKPOINT2 \
#   experiment_name=sim2drive_virtual_only \
#   metric_cache_path=$CACHE_PATH \
#   original_sensor_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs \
#   synthetic_sensor_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs \
#   synthetic_scenes_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles \
#   +max_scenarios=100