#!/bin/bash
# 增强剪枝策略运行示例
# 展示如何使用新的 cosine + plateau 调度和 HardConcrete 温度退火

set -e

# 设置环境
source /opt/miniconda3/bin/activate wedefense
export PYTHONPATH=/Users/pengjy/wespeaker_hubert:$PYTHONPATH

# 配置参数
CONFIG_FILE="conf/enhanced_pruning_example.yaml"
EXP_DIR="exp/enhanced_pruning_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="/path/to/your/data"  # 请替换为实际数据路径

echo "开始增强剪枝训练..."
echo "配置文件: $CONFIG_FILE"
echo "实验目录: $EXP_DIR"
echo "数据目录: $DATA_DIR"

# 运行训练（需要根据实际环境调整）
python -m wespeaker.bin.train_pq \
    --config $CONFIG_FILE \
    --exp_dir $EXP_DIR \
    --data_dir $DATA_DIR \
    --use_pruning_loss true \
    --sparsity_schedule cosine_plateau \
    --plateau_start_ratio 0.9 \
    --use_plateau_lr_decay true \
    --lr_decay_factor 0.1 \
    --min_lr_ratio 0.01 \
    --num_epochs 100 \
    --sparsity_warmup_epochs 20 \
    --target_sparsity 0.7

echo "训练完成！"
echo "实验目录: $EXP_DIR"
echo "可以使用以下命令进行推理："
echo "python -m wespeaker.bin.extract --config $CONFIG_FILE --model_path $EXP_DIR/models/final.pt"
