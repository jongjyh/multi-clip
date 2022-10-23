# ladder
/home/chenzhongzhi/clash/clash-linux-amd64-v1.11.4 -d /home/chenzhongzhi/clash  &>/home/chenzhongzhi/clash/network.log & 
sleep 2s
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=sock5://127.0.0.1:7891

# exp
lr=5e-5
wd=1e-1
ep=10
seed=42
loss_fn=mse
pooler_fn=cls
layer_kd=false
task=multi-clip
student=xlm-roberta-base
# student=hfl/chinese-roberta-wwm-ext
teacher=openai/clip-vit-large-patch14
# teacher=openai/clip-vit-base-patch16
alpha=.1
# dataset path
dst=/sharefs/czz/datasets/multi-clip/cc3m-zh
# dst=/sharefs/czz/datasets/laion28m
bs=256
gpus=4
warmup_steps=1000
kd_type=kd
# run_name is also output path
run_name=debug_test


# WANDB_PROJECT=clip-kd HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python /home/chenzhongzhi/multi-clip/multi-clip/run_translation.py  \
WANDB_PROJECT=bilingual-kd HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python  \
     /home/chenzhongzhi/multi-clip/multi-clip/run_translation.py  \
    --model_name_or_path ${student} \
    --do_train \
    --do_eval \
    --max_train_samples 100000 \
    --warmup_steps ${warmup_steps} \
    --source_lang zh \
    --target_lang en \
    --max_source_length 50 \
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
    --output_dir ckpt/${run_name} \
    --dataset_name $dst \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 512 \
    --metric_for_best_model eval_cossim_loss \
    --greater_is_better false \
    --loss_fn ${loss_fn} \
    --pooler_fn ${pooler_fn}  \
    --layer_kd ${layer_kd} \
    --teacher_model ${teacher} \
    --load_best_model_at_end \
    --alpha ${alpha} \
    --kd_type ${kd_type} \
    