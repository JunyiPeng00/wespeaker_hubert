#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
# 修复版本：解决multiprocessing fork()警告

. ./path.sh || exit 1
. ./lumi.sh || exit 1

# 抑制multiprocessing警告
export PYTHONWARNINGS="ignore::DeprecationWarning"
export OMP_NUM_THREADS=1

stage=3
stop_stage=6

HOST_NODE_ADDR="localhost:29400"
num_nodes=1
job_id=2024

data=data
data_type="raw"  # shard/raw

config=conf/resnet.yaml
ft_config=conf/resnet.yaml
exp_dir=exp/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150
ft_exp_dir=exp/ft
gpus="[0,1,2,3,4,5,6,7]"
num_avg=3
checkpoint=

trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

# setup for large margin fine-tuning
lm_config=conf/resnet_lm.yaml

train_data=vox2_dev
test_data=vox1

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --stage 2 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in vox2_dev vox1; do
    if [ $data_type == "shard" ]; then
      python -W ignore::DeprecationWarning tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 16 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python -W ignore::DeprecationWarning tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done
  # Convert all musan data to LMDB
  python -W ignore::DeprecationWarning tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # Convert all rirs data to LMDB
  python -W ignore::DeprecationWarning tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  
  # 使用torchrun替代torch.distributed.launch (推荐方法)
  if command -v torchrun &> /dev/null; then
    echo "使用torchrun启动训练..."
    torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
      wespeaker/bin/train_pq.py --config $config \
        --exp_dir ${exp_dir} \
        --gpus $gpus \
        --num_avg ${num_avg} \
        --data_type "${data_type}" \
        --train_data ${data}/vox2_dev/${data_type}.list \
        --train_label ${data}/vox2_dev/utt2spk \
        --train_lmdb ${data}/vox2_dev/lmdb \
        --reverb_data ${data}/rirs/lmdb \
        --noise_data ${data}/musan/lmdb \
        ${checkpoint:+--checkpoint $checkpoint}
  else
    echo "使用torch.distributed.launch启动训练..."
    python -W ignore::DeprecationWarning -m torch.distributed.launch --standalone --nnodes=1 --nproc_per_node=$num_gpus \
      wespeaker/bin/train_pq.py --config $config \
        --exp_dir ${exp_dir} \
        --gpus $gpus \
        --num_avg ${num_avg} \
        --data_type "${data_type}" \
        --train_data ${data}/vox2_dev/${data_type}.list \
        --train_label ${data}/vox2_dev/utt2spk \
        --train_lmdb ${data}/vox2_dev/lmdb \
        --reverb_data ${data}/rirs/lmdb \
        --noise_data ${data}/musan/lmdb \
        ${checkpoint:+--checkpoint $checkpoint}
  fi
fi

# 其余阶段保持不变...
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  . ./lumi.sh || exit 1
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python -W ignore::DeprecationWarning wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}

  model_path=$avg_model
  pru_model_path=$model_path
  pru_config=${exp_dir}/config.yaml

  echo "Extract embeddings ..."
  local/extract_vox_test.sh \
    --exp_dir $exp_dir --model_path $pru_model_path --config_path $pru_config \
    --nj 8 --gpus $gpus --data_type $data_type --data ${data} \
    --data_train ${train_data}  \
    --data_test ${test_data}
    
  local/extract_vox_train.sh \
    --exp_dir $exp_dir --model_path $pru_model_path --config_path $pru_config \
    --nj 16 --gpus $gpus --data_type $data_type --data ${data} \
    --data_train ${train_data}  
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score norm ..."
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Score calibration ..."
  local/score_calibration.sh \
    --stage 1 --stop-stage 5 \
    --score_norm_method $score_norm_method \
    --calibration_trial "vox2_cali.kaldi" \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Full fine-tuning ..."
  mkdir -p ${ft_exp_dir}/models
  cp ${exp_dir}/models/avg_model.pt ${ft_exp_dir}/models/model_0.pt
  bash ./run_wavlm_ori_fixed.sh --stage 3 --stop_stage 7 \
      --data ${data} \
      --data_type ${data_type} \
      --config ${ft_config} \
      --exp_dir ${ft_exp_dir} \
      --train_data ${train_data} \
      --test_data ${test_data} \
      --gpus $gpus \
      --num_avg 1 \
      --checkpoint ${ft_exp_dir}/models/model_0.pt \
      --trials "$trials" \
      --score_norm_method ${score_norm_method} \
      --top_n ${top_n}
fi
