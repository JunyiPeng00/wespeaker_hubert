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
    mhfa_WavLMBasePlus
    mhfa_WavLMBasePlus_w8
    mhfa_WavLMBasePlus_p70_e
    mhfa_WavLMBasePlus_p70_qua
    mhfa_WavLMBasePlus_p70_qua
    s70/mhfa_WavLMBasePlus_w4
    s70/mhfa_WavLMBasePlus_w8
)
config=${configs[$SLURM_ARRAY_TASK_ID-1]} 

# singularity exec $SIFPYTORCH bash run_wavlm_ori_cn.sh \
#     --config conf/cnceleb/baseline/${config}_frozen.yaml \
#     --exp_dir exp/cnceleb/baseline/${config}-frozen \
#     --ft_config conf/cnceleb/baseline/${config}_ft.yaml \
#     --ft_exp_dir exp/cnceleb/baseline/${config}-ft \
#     --stage 8 --stop_stage 8

# singularity exec $SIFPYTORCH bash run_wavlm_ori_cn.sh \
#     --config conf/cnceleb/baseline/mhfa_WavLMBasePlus_frozen.yaml \
#     --exp_dir exp/cnceleb/baseline/mhfa_WavLMBasePlus-frozen \
#     --ft_config conf/cnceleb/qua/${config}.yaml \
#     --ft_exp_dir exp/cnceleb/qua/${config} \
#     --stage 8 --stop_stage 8

# singularity exec $SIFPYTORCH bash run_wavlm_cn_pruning.sh \
#     --config conf/cnceleb/baseline/mhfa_WavLMBasePlus_frozen.yaml \
#     --exp_dir exp/cnceleb/baseline/mhfa_WavLMBasePlus-frozen \
#     --ft_config conf/cnceleb/pruning_qua/${config}.yaml \
#     --ft_exp_dir exp/cnceleb/pruning_qua/${config} \
#     --stage 8 --stop_stage 8

singularity exec $SIFPYTORCH bash run_wavlm_cn_pruning.sh \
    --config conf/cnceleb/qua/mhfa_WavLMBasePlus_w8.yaml \
    --exp_dir exp/cnceleb/qua/mhfa_WavLMBasePlus_w8 \
    --ft_config conf/cnceleb/qua2pruning/${config}.yaml \
    --ft_exp_dir exp/cnceleb/qua2pruning/${config} \
    --stage 8 --stop_stage 8

# singularity exec $SIFPYTORCH bash run_wavlm_ori_cn_p2q.sh \
#     --config exp/cnceleb/pruning/mhfa_WavLMBasePlus_p70_e/config_pruned.yaml \
#     --exp_dir exp/cnceleb/pruning/mhfa_WavLMBasePlus_p70_e \
#     --ft_config conf/cnceleb/pruning2qua/${config}.yaml \
#     --ft_exp_dir exp/cnceleb/pruning2qua/${config} \
#     --stage 8 --stop_stage 8