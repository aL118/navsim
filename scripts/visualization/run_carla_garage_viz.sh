#!/bin/bash

# SLURM script to generate simulator footage visualizations for CarlaGarage

#SBATCH --job-name=viz_carla_garage
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/navsim/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/navsim/my_dump/%x.out.%j

## Resource allocation
#SBATCH --mem=64gb                                               # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --ntasks=4

## Time and partition config
#SBATCH --time=4:00:00
#SBATCH --qos=medium
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

TRAIN_TEST_SPLIT=navtrain  # Use test split for real-world performance
CHECKPOINT=/fs/nexus-scratch/aliu1237/carla_garage_clone/logs/sim2drive_stage2_19_30/model_0029.pth
CACHE_PATH=/fs/nexus-projects/sim2real/aliu/navsim/metric_cache
EXPERIMENT="carla_garage_sim2real_visualizations"
VIZ_OUTPUT_DIR=/fs/nexus-projects/sim2real/aliu/navsim/my_dump/$EXPERIMENT
MAX_VIZ_SCENARIOS=10
VIZ_FPS=10
VIZ_FORMAT=gif

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_with_viz.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=carla_garage_agent \
worker=single_machine_thread_pool \
agent.checkpoint_path=$CHECKPOINT \
train_test_split.scene_filter.num_history_frames=20 \
train_test_split.scene_filter.num_future_frames=20 \
experiment_name=$EXPERIMENT \
metric_cache_path=$CACHE_PATH \
+viz_output_dir=$VIZ_OUTPUT_DIR \
+max_viz_scenarios=$MAX_VIZ_SCENARIOS \
+viz_fps=$VIZ_FPS \
+viz_format=$VIZ_FORMAT \

# Usage:
# sbatch scripts/visualization/run_carla_garage_viz.sh
