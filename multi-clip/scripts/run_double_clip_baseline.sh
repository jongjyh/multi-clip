# ladder
source /home/chenzhongzhi/proxy.sh

# exp
lr=5e-5
wd=1e-1
ep=10
seed=42
loss_fn=mse
task=multi-clip
student=xlm-roberta-base
teacher=openai/clip-vit-large-patch14
# dataset path
dst=/sharefs/czz/datasets/multi-clip/cc3m-zh
# dst=/sharefs/czz/datasets/laion28m
bs=256
warmup_steps=1000
run_name=double_clip_xlmBase_p14_bs${bs}_wd${wd}_lr${lr}_ep${ep}_ws${warmup_steps}_baseline_100k
# run_name=xlm_base_${gpus}_${loss_fn}_${pooler_fn}_wd${wd}_bs${bs}_lr${lr}_warm${warmup_steps}_ep${ep}_sd${seed}_${kd_type}_enandzh_cc3m
debug=0
baseline=true

# WANDB_PROJECT=clip-kd HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python /home/chenzhongzhi/multi-clip/multi-clip/run_translation.py  \
CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=double-clip HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python \
    /home/chenzhongzhi/multi-clip/multi-clip/run_kd.py  \
    --model_name_or_path ${student} \
    --do_train \
    --do_eval \
    --max_train_samples 100000 \
    --warmup_steps ${warmup_steps} \
    --source_lang zh \
    --target_lang en \
    --max_source_length 75 \
    --num_train_epochs $ep \
    --remove_unused_columns false \
    --weight_decay $wd \
    --learning_rate $lr \
    --seed $seed \
    --report_to wandb \
    --evaluation_strategy steps \
    --save_total_limit 1 \
    --run_name ${run_name} \
    --logging_steps 500 \
    --output_dir ckpt/${run_name} \
    --dataset_name $dst \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 256 \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --loss_fn ${loss_fn} \
    --teacher_model ${teacher} \
    --baseline ${baseline} \
    
    