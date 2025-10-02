#!/bin/bash
# 测试修复后的推理脚本
# 应该看到: "✅ Checkpoint already contains quantization parameters"
# 并且EER应该显著降低 (< 5%)

echo "################################################################################"
echo "  🧪 Testing Fixed Inference Script"
echo "################################################################################"
echo ""

WORK_DIR=/scratch/project_465002053/junyi/sv/wespeaker_dev/wespeaker_hubert/examples/voxceleb/v4_pruning
cd $WORK_DIR

. ./path.sh || exit 1

# 配置
EXP_DIR="exp/qua_v2/mhfa_WavLMBasePlus_w8"
CONFIG="${EXP_DIR}/config.yaml"
MODEL="${EXP_DIR}/models/avg_model.pt"

echo "📁 Experiment: ${EXP_DIR}"
echo "📄 Config: ${CONFIG}"
echo "💾 Model: ${MODEL}"
echo ""

# 加载环境
module load LUMI PyTorch/2.2.2-rocm-5.6.1-python-3.10-singularity-20240617

echo "=================================="
echo "Step 1: Clean old embeddings"
echo "=================================="
rm -rf ${EXP_DIR}/embeddings/vox1_test
echo "✅ Cleaned"
echo ""

echo "=================================="
echo "Step 2: Extract embeddings (FIXED)"
echo "=================================="
echo "Expected output:"
echo "  - '✅ Checkpoint already contains quantization parameters'"
echo "  - 'Skipping quantization application'"
echo ""

# 提取embeddings（只测试前100个utterance）
singularity exec $SIFPYTORCH bash << 'SINGULARITY_EOF'
cd /scratch/project_465002053/junyi/sv/wespeaker_dev/wespeaker_hubert/examples/voxceleb/v4_pruning

# 创建临时测试数据（只取前100条）
TEST_LIST="data/vox1/raw.list"
head -100 ${TEST_LIST} > /tmp/vox1_test_100.list

# 提取embeddings
CUDA_VISIBLE_DEVICES=0 python wespeaker/bin/extract.py \
  --config exp/qua_v2/mhfa_WavLMBasePlus_w8/config.yaml \
  --model_path exp/qua_v2/mhfa_WavLMBasePlus_w8/models/avg_model.pt \
  --data_type raw \
  --data_list /tmp/vox1_test_100.list \
  --embed_ark exp/qua_v2/mhfa_WavLMBasePlus_w8/embeddings/vox1_test/xvector.ark \
  --batch-size 1 \
  --num-workers 1 \
  --train_lmdb data/vox1/lmdb \
  2>&1 | tee /tmp/extract_test.log

echo ""
echo "=================================="
echo "Extraction Log Analysis:"
echo "=================================="
if grep -q "✅ Checkpoint already contains quantization parameters" /tmp/extract_test.log; then
    echo "✅ GOOD: Detected quantization in checkpoint"
else
    echo "❌ BAD: Did not detect quantization (check logs)"
fi

if grep -q "Skipping quantization application" /tmp/extract_test.log; then
    echo "✅ GOOD: Skipped redundant quantization"
else
    echo "❌ BAD: May have applied quantization again"
fi

if grep -q "Quantization applied successfully" /tmp/extract_test.log; then
    echo "❌ WARNING: Should NOT see 'Quantization applied successfully'"
    echo "   This means quantization was applied again (wrong!)"
fi

SINGULARITY_EOF

echo ""
echo "=================================="
echo "Step 3: Check Results"
echo "=================================="

if [ -f "${EXP_DIR}/embeddings/vox1_test/xvector.scp" ]; then
    NUM_EMBEDS=$(wc -l < ${EXP_DIR}/embeddings/vox1_test/xvector.scp)
    echo "✅ Extraction successful: ${NUM_EMBEDS} embeddings"
    echo ""
    echo "📊 Next Steps:"
    echo "   1. Run full extraction: bash run_wavlm_ori.sh --stage 4 --stop_stage 4"
    echo "   2. Compute scores: bash run_wavlm_ori.sh --stage 5 --stop_stage 5"
    echo "   3. Verify EER drops to < 5%"
else
    echo "❌ Extraction failed - check logs"
fi

echo ""
echo "=================================="
echo "Quick Embedding Sanity Check"
echo "=================================="
echo "Checking if embeddings have reasonable statistics..."

singularity exec $SIFPYTORCH python << 'PYEOF'
import kaldiio
import numpy as np

scp_path = "exp/qua_v2/mhfa_WavLMBasePlus_w8/embeddings/vox1_test/xvector.scp"
try:
    embeddings = []
    for key, emb in kaldiio.load_scp(scp_path):
        embeddings.append(emb)
        if len(embeddings) >= 10:  # 只检查前10个
            break
    
    embeddings = np.vstack(embeddings)
    print(f"Shape: {embeddings.shape}")
    print(f"Mean: {embeddings.mean():.4f}")
    print(f"Std: {embeddings.std():.4f}")
    print(f"Min: {embeddings.min():.4f}")
    print(f"Max: {embeddings.max():.4f}")
    
    # 检查异常值
    if np.abs(embeddings.mean()) > 10 or embeddings.std() < 0.01 or embeddings.std() > 100:
        print("\n❌ WARNING: Embedding statistics look abnormal!")
        print("   Model may still have issues.")
    else:
        print("\n✅ Embedding statistics look reasonable!")
        print("   Fix is likely working correctly.")
except Exception as e:
    print(f"Could not check embeddings: {e}")
PYEOF

echo ""
echo "################################################################################"
echo "  📋 Summary"
echo "################################################################################"
echo ""
echo "If you saw:"
echo "  ✅ 'Checkpoint already contains quantization parameters'"
echo "  ✅ 'Skipping quantization application'"
echo "  ✅ Reasonable embedding statistics"
echo ""
echo "Then the fix is working! Proceed with full evaluation."
echo ""
echo "Full evaluation command:"
echo "  cd ${WORK_DIR}"
echo "  singularity exec \$SIFPYTORCH bash run_wavlm_ori.sh \\"
echo "      --config ${CONFIG} \\"
echo "      --exp_dir ${EXP_DIR} \\"
echo "      --stage 4 --stop_stage 6"
echo ""

