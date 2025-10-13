alias check='df -h .'
alias gpu='nvidia-smi -L'
alias jobs='squeue -u aliu1237 -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"'
alias inquire='squeue -w gammagpu[00-21]'

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"