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
    # MHFA_large-frozen_v1
    MHFA_large-ft-s10
    MHFA_large-ft-s20
    MHFA_large-ft-s30
    MHFA_large-ft-s40
    MHFA_large-ft-s50
    MHFA_large-ft-s60
    MHFA_large-ft-s70
    MHFA_large-ft-s80
    MHFA_large-ft-s90
    MHFA_WavLM_Base_Plus-s10
    MHFA_WavLM_Base_Plus-s20
    MHFA_WavLM_Base_Plus-s30
    MHFA_WavLM_Base_Plus-s40
    MHFA_WavLM_Base_Plus-s50
    MHFA_WavLM_Base_Plus-s60
    MHFA_WavLM_Base_Plus-s70
    MHFA_WavLM_Base_Plus-s80
    MHFA_WavLM_Base_Plus-s90
    # MHFA_WavLM_Base_Plus-s10
    # MHFA_WavLM_Base_Plus-ft-s10-2
    # MHFA_WavLM_Base_Plus-ft-s20-2
    # MHFA_WavLM_Base_Plus-ft-s30-2
    # MHFA_WavLM_Base_Plus-ft-s40-2
    # MHFA_WavLM_Base_Plus-ft-s50-2
    # MHFA_WavLM_Base_Plus-ft-s60-2
    # MHFA_WavLM_Base_Plus-ft-s70-2
    # MHFA_WavLM_Base_Plus-ft-s80-2
    # MHFA_WavLM_Base_Plus-ft-s90-2
    # CA-MHFA_WavLM-Base-Plus_VoxCeleb2
    # MHFA_WavLM_Base_Plus-frozen
    # MHFA_WavLM_Base_Plus-ft
    # MHFA_WavLM_Base_Plus-ft-s10
    # MHFA_WavLM_Base_Plus-ft-s30
    # MHFA_WavLM_Base_Plus-ft-s50
    # MHFA_WavLM_Base_Plus-ft-s70
    # MHFA_WavLM_Base_Plus-ft-s90
    # MHFA_WavLM_Base_Plus-ft-s50-lr1
    # MHFA_WavLM_Base_Plus-ft-s50-lr2
)
config=${configs[$SLURM_ARRAY_TASK_ID-1]} 

# singularity exec $SIFPYTORCH bash run_wavlm_ori.sh \
#     --config conf/training_purning_large/${config}.yaml \
#     --exp_dir exp/training_purning_large/${config} \
#     --stage 3 --stop_stage 7

# singularity exec $SIFPYTORCH bash run_wavlm_ori_eval.sh \
#     --config conf/training_purning_large/${config}.yaml \
#     --exp_dir exp/training_purning_large/${config} \
#     --stage 4 --stop_stage 7

# singularity exec $SIFPYTORCH bash run_wavlm_ori.sh \
    # --config conf/training_purning_large/MHFA_large-frozen_v1.yaml \
    # --exp_dir exp/training_purning_large/MHFA_large-frozen_v1 \
    # --ft_config conf/training_purning_large/${config}.yaml \
    # --ft_exp_dir exp/training_purning_large/${config} \
    # --stage 8 --stop_stage 8


singularity exec $SIFPYTORCH bash run_wavlm_pruning.sh \
    --config conf/training_purning_large/MHFA_large-frozen_v1.yaml \
    --exp_dir exp/training_purning_large/MHFA_large-frozen_v1 \
    --ft_config conf/training_purning_large_v6/${config}.yaml \
    --ft_exp_dir exp/training_purning_large_v6/${config} \
    --stage 8 --stop_stage 8

# singularity exec $SIFPYTORCH bash run_wavlm_pruning_eval.sh \
#     --config conf/MHFA_WavLM_Base_Plus-frozen.yaml \
#     --exp_dir exp/MHFA_WavLM_Base_Plus-frozen \
#     --ft_config conf/${config}.yaml \
#     --ft_exp_dir exp/${config} \
#     --stage 8 --stop_stage 8


# singularity exec $SIFPYTORCH bash run_wavlm_pruning.sh \
#     --config conf/MHFA_WavLM_Base_Plus-frozen.yaml \
#     --exp_dir exp/MHFA_WavLM_Base_Plus-frozen \
#     --ft_config conf/${config}.yaml \
#     --ft_exp_dir exp/${config} \
#     --stage 8 --stop_stage 8

# singularity exec $SIFPYTORCH bash run_wavlm_ori.sh \
#     --config conf/VoxCeleb_s2/${config}-frozen.yaml  \
#     --exp_dir exp/VoxCeleb_s2/${config}-frozen \
#     --ft_config conf/VoxCeleb_s2/${config}-ft.yaml \
#     --ft_exp_dir exp/VoxCeleb_s2/${config}-ft \
#     --stage 3 --stop_stage 8