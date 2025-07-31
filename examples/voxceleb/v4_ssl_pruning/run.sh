#!/bin/bash

time=$(date +"%Y-%m-%d")
echo "Date: $time"

WORK_DIR=/scratch/project_465001402/junyi/sv/ICASSP26_Pruning/wespeaker_hubert/examples/voxceleb/v4_ssl_pruning/
cd $WORK_DIR

my_folder="${WORK_DIR}/log/${time}/"

mkdir -p "${my_folder}"

sbatch -J "WeSpk_Pruning" \
  --time "24:00:00" \
  --array "1-9" \
  -o "${my_folder}/output_%x_%j_%a.txt" \
  -e "${my_folder}/error_%x_%j_%a.txt" \
    submit_vox2.sh


# sbatch -J "WeSpk_Pruning_CN" \
#   --time "24:00:00" \
#   --array "1-1" \
#   -o "${my_folder}/output_%x_%j_%a.txt" \
#   -e "${my_folder}/error_%x_%j_%a.txt" \
#     submit_cn.sh