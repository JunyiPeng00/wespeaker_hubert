#!/bin/bash

time=$(date +"%Y-%m-%d")
echo "Date: $time"

WORK_DIR=/scratch/project_465002053/junyi/sv/wespeaker_dev/wespeaker_hubert/examples/voxceleb/v4_pruning
cd $WORK_DIR

my_folder="${WORK_DIR}/log/${time}/"

mkdir -p "${my_folder}"

sbatch -J "WeSpk_Pruning_CN" \
  --time "24:00:00" \
  --array "4-4" \
  -o "${my_folder}/output_%x_%j_%a.txt" \
  -e "${my_folder}/error_%x_%j_%a.txt" \
    submit_cn.sh

# sbatch -J "WeSpk_Qua_Vox2" \
#   --time "24:00:00" \
#   --array "7-7" \
#   -o "${my_folder}/output_%x_%j_%a.txt" \
#   -e "${my_folder}/error_%x_%j_%a.txt" \
#     submit_vox2_qua.sh

# sbatch -J "WeSpk_PruningQua_Vox2" \
#   --time "24:00:00" \
#   --array "1-8" \
#   -o "${my_folder}/output_%x_%j_%a.txt" \
#   -e "${my_folder}/error_%x_%j_%a.txt" \
#     submit_vox2_pruning_qua.sh

# sbatch -J "WeSpk_PruningQua_Vox2" \
#   --time "24:00:00" \
#   --array "7-7" \
#   -o "${my_folder}/output_%x_%j_%a.txt" \
#   -e "${my_folder}/error_%x_%j_%a.txt" \
#     submit_vox2_pruning.sh