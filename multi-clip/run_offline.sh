# ladder
/sharefs/czz/clash/clash-linux-amd64-v1.11.4 -d /sharefs/czz/clash  &>/dev/null & 
sleep 1s
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=sock5://127.0.0.1:7891

# exp
lr=2e-5
wd=1e-4
ep=5
seed=42
loss_fn=mse
pooler_fn=average
layer_kd=false
task=multi-clip
student=hfl/chinese-roberta-wwm-ext
teacher=openai/clip-vit-large-patch14
dst=/home/chenzhongzhi/czz/datasets/multi-clip/cc100k-zh 
run_name=baai/${student}_${teacher}_${loss_fn}_${pooler_fn}_lkd${layer_kd}_nolinear_losssum_wd${wd}_lr${lr}_ep${ep}_sd${seed}_dst${dst}
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python /home/chenzhongzhi/repo/multi-clip/multi-clip/run_translation.py  \
    --model_name_or_path ${student} \
    --do_train \
    --do_eval \
    --source_lang zh \
    --target_lang en \
    --max_source_length 75 \
    --num_train_epochs $ep \
    --weight_decay $wd \
    --learning_rate $lr \
    --seed $seed \
    --pad_to_max_length true \
    --report_to wandb \
    --evaluation_strategy steps \
    --save_total_limit 1 \
    --run_name ${run_name} \
    --logging_steps 500 \
    --output_dir ckpt/${task}/${run_name} \
    --dataset_name $dst \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --loss_fn ${loss_fn} \
    --pooler_fn ${pooler_fn}  \
    --layer_kd ${layer_kd} \
    --overwrite_output_dir \
    --teacher_model ${teacher} \
