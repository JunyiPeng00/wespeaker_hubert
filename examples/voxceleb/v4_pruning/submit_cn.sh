#!/bin/bash

#SBATCH -J WeSpk_Pruning
#SBATCH -p standard-g
#SBATCH --gres=gpu:8
#SBATCH --account=project_465002053
#SBATCH --cpus-per-task=56
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=47:59:00
#SBATCH --output=log/output_%x_%j.txt
#SBATCH --error=log/error_%x_%j.txt
#SBATCH --array=1-9

module load LUMI PyTorch/2.2.2-rocm-5.6.1-python-3.10-singularity-20240617



configs=(
    MHFA_WavLM_Base_Plus-ft
    MHFA_WavLM_Base_Plus-ft-s10-2
    MHFA_WavLM_Base_Plus-ft-s20-2
    MHFA_WavLM_Base_Plus-ft-s30-2
    MHFA_WavLM_Base_Plus-ft-s40-2
    MHFA_WavLM_Base_Plus-ft-s50-2
    MHFA_WavLM_Base_Plus-ft-s60-2
    MHFA_WavLM_Base_Plus-ft-s70-2
    MHFA_WavLM_Base_Plus-ft-s80-2
    MHFA_WavLM_Base_Plus-ft-s90-2
)
config=${configs[$SLURM_ARRAY_TASK_ID-1]} 

# singularity exec $SIFPYTORCH bash run_wavlm_ori_cn.sh \
#     --config conf/CNCeleb_s2/${config}-frozen.yaml \
#     --exp_dir exp/CNCeleb_s2/${config}-frozen \
#     --ft_config conf/CNCeleb_s2/${config}-ft.yaml \
#     --ft_exp_dir exp/CNCeleb_s2/${config}-ft \
#     --stage 8 --stop_stage 8

# singularity exec $SIFPYTORCH bash run_wavlm_ori_cn.sh \
#     --config conf/CNCeleb/MHFA_WavLM_Base_Plus-frozen.yaml \
#     --exp_dir exp/CNCeleb/MHFA_WavLM_Base_Plus-frozen \
#     --ft_config conf/CNCeleb/${config}.yaml \
#     --ft_exp_dir exp/CNCeleb/${config} \
#     --stage 8 --stop_stage 8

# singularity exec $SIFPYTORCH bash run_wavlm_cn_pruning.sh \
#     --config conf/CNCeleb/MHFA_WavLM_Base_Plus-frozen.yaml \
#     --exp_dir exp/CNCeleb/MHFA_WavLM_Base_Plus-frozen \
#     --ft_config conf/CNCeleb/${config}.yaml \
#     --t_exp_dir exp/CNCeleb/${config} \
#     --stage 8 --stop_stage 8

singularity exec $SIFPYTORCH bash run_wavlm_ori_cn.sh \
    --config conf/CNCeleb/${config}.yaml \
    --exp_dir exp/${config} \
    --stage 4 --stop_stage 7