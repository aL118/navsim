#!/bin/bash

# SLURM script to evaluate transfuser on a small subset (100 scenarios)

#SBATCH --job-name=metrics_subset
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/navsim/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/navsim/my_dump/%x.out.%j

## Resource allocation
#SBATCH --mem=64gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --ntasks=4

## Time and partition config
#SBATCH --time=2:00:00
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

# Use navmini for a smaller subset (or specify max_scenarios)
TRAIN_TEST_SPLIT=navtrain
CHECKPOINT=/fs/nexus-projects/sim2real/aliu/navsim/models/carla_garage/pretrained_baseline_0030_0.ckpt
CACHE_PATH=/fs/nexus-projects/sim2real/aliu/navsim/metric_cache

# Use run_pdm_score.py (the evaluation script without visualization)
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
worker=single_machine_thread_pool \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=carla_garage_subset \
metric_cache_path=$CACHE_PATH \
+max_scenarios=100

# Usage:
# sbatch scripts/evaluation/run_transfuser_metrics_subset.sh
