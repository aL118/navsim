#!/bin/bash

#SBATCH --job-name=metric_cache
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/navsim/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/navsim/my_dump/%x.out.%j

#SBATCH --mem=120gb
#SBATCH --ntasks=4
#SBATCH --time=48:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

eval "$(conda shell.bash hook)"
conda activate navsim

export NAVSIM_DEVKIT_ROOT="/fs/nexus-projects/sim2real/aliu/navsim"
export NAVSIM_EXP_ROOT="/fs/nexus-projects/sim2real/aliu/navsim/exp"
export OPENSCENE_DATA_ROOT="/fs/nexus-projects/sim2real/aliu/navsim/dataset"

export NUPLAN_MAPS_ROOT="/fs/nexus-projects/sim2real/aliu/navsim/dataset/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

TRAIN_TEST_SPLIT=navhard_two_stage  # Generate cache for navhard test split
CACHE_PATH=/fs/nexus-projects/sim2real/aliu/navsim/metric_cache_navhard

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
metric_cache_path=$CACHE_PATH

# sbatch -J metric_cache scripts/evaluation/run_metric_caching_slurm.sh