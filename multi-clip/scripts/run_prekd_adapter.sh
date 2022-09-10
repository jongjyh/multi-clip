# ladder
/sharefs/czz/clash/clash-linux-amd64-v1.11.4 -d /sharefs/czz/clash  &>/dev/null & 
sleep 1s
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=sock5://127.0.0.1:7891

# exp
lr=8e-4
wd=1e-4
ep=5
seed=42
loss_fn=mse
pooler_fn=average
layer_kd=false
task=multi-clip
student=hfl/chinese-roberta-wwm-ext
# student=hfl/chinese-roberta-wwm-ext
# teacher=openai/clip-vit-large-patch14
teacher=openai/clip-vit-base-patch32
alpha=.1
# dst=/sharefs/czz/datasets/mt/merge_cc3m_tsl2019
dst=/sharefs/czz/datasets/multi-clip/cc3m-zh
bs=512
# dst=/home/chenzhongzhi/czz/datasets/multi-clip/cc100k-zh 
gpus=2

warmup_steps=1000
kd_type=prekd
delta=adapter
run_name=robBase_vitBase32_${gpus}_${loss_fn}_${pooler_fn}_wd${wd}_bs${bs}_lr${lr}_warm${warmup_steps}_ep${ep}_sd${seed}_${kd_type}_${delta}

WANDB_PROJECT=clip-kd HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m torch.distributed.launch \
    --nproc_per_node $gpus /home/chenzhongzhi/multi-clip/multi-clip/run_translation.py  \
    --model_name_or_path ${student} \
    --do_train \
    --do_eval \
    --warmup_steps ${warmup_steps} \
    --fp16 \
    --source_lang zh \
    --target_lang en \
    --max_source_length 30 \
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
    --delta ${delta} \
    --overwrite_output_dir true \
