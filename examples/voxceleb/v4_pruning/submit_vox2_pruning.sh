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
#SBATCH --array=1-1

module load LUMI PyTorch/2.2.2-rocm-5.6.1-python-3.10-singularity-20240617
# module load LUMI PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527



configs=(
    mhfa_WavLMBasePlus_p70
    mhfa_WavLMBasePlus_p70_s1
    mhfa_WavLMBasePlus_p70_s2
    mhfa_WavLMBasePlus_p70_s3
    mhfa_WavLMBasePlus_p70_s4
    mhfa_WavLMBasePlus_p70_e
    mhfa_WavLMBasePlus_p70_qua
)
config=${configs[$SLURM_ARRAY_TASK_ID-1]} 


# singularity exec $SIFPYTORCH bash run_wavlm_pruning.sh \
#     --config conf/baseline_vox2/mhfa_WavLMBasePlus_frozen.yaml \
#     --exp_dir exp/baseline_vox2/mhfa_WavLMBasePlus_frozen \
#     --ft_config conf/pruning/${config}.yaml \
#     --ft_exp_dir exp/pruning/${config} \
#     --stage 8 --stop_stage 8

# singularity exec $SIFPYTORCH bash run_wavlm_pruning_eval.sh \
#     --config conf/baseline_vox2/mhfa_WavLMBasePlus_frozen.yaml \
#     --exp_dir exp/baseline_vox2/mhfa_WavLMBasePlus_frozen \
#     --ft_config conf/pruning/${config}.yaml \
#     --ft_exp_dir exp/pruning/${config} \
#     --stage 8 --stop_stage 8

singularity exec $SIFPYTORCH bash run_wavlm_pruning.sh \
    --config conf/qua/mhfa_WavLMBasePlus_w8.yaml \
    --exp_dir exp/qua/mhfa_WavLMBasePlus_w8 \
    --ft_config conf/qua2pruning/${config}.yaml \
    --ft_exp_dir exp/qua2pruning/${config} \
    --stage 8 --stop_stage 8

# singularity exec $SIFPYTORCH bash run_wavlm_pruning.sh \
#     --config conf/training_purning_length/${config}.yaml \
#     --exp_dir exp/training_purning_length/${config} \
#     --stage 3 --stop_stage 7

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