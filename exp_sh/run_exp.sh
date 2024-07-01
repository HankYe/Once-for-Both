#!/bin/bash
echo "Start Searching"
cd ../
n_gpu=2
gpu=0,1
master_port=1235
model_name=deit_small_patch16_224_mim
data_path=/path/to/imagenet/
output_dir=runs/exp
target_flops=1.0
batch_size=128
eff_bs=1024
accum_iter=`expr $eff_bs / $batch_size / $n_gpu`
mkdir -p $output_dir
python -m torch.distributed.launch --nproc_per_node $n_gpu --master_port $master_port --use_env search.py --model $model_name --output_dir $output_dir --target_flops $target_flops --gpu $gpu --attn_search --mlp_search --embed_search --mae --batch-size $batch_size --accum-iter $accum_iter --data-path $data_path 2>&1 | tee "$output_dir/Search.log"
echo "Start Fusing"
python -m torch.distributed.launch --nproc_per_node $n_gpu --master_port $master_port --use_env search.py --model $model_name --output_dir $output_dir --target_flops $target_flops --gpu $gpu --attn_search --mlp_search --embed_search --mae --batch-size $batch_size --accum-iter $accum_iter --data-path $data_path --resume --checkpoint "$output_dir/model_fused.pth" 2>&1 | tee "$output_dir/Search_resume_fused.log"

echo "Start Finetuning"
model_name=deit_small_patch16_224_finetune
mkdir -p "${output_dir}_finetune/"
python -m torch.distributed.launch --nproc_per_node $n_gpu --master_port $master_port --use_env finetune.py --model $model_name --output_dir "${output_dir}_finetune/" --gpu $gpu --batch-size $batch_size --accum-iter $accum_iter --finetune "${output_dir}/best.pth" --data-path $data_path 2>&1 | tee "${output_dir}_finetune/Finetune.log"
