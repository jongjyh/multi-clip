# exp
lr=1e-4
wd=1e-1
ep=10
seed=42
loss_fn=mse
pooler_fn=cls
layer_kd=false
task=multi-clip
student=hfl/chinese-roberta-wwm-ext-large
teacher=openai/clip-vit-large-patch14
# teacher=openai/clip-vit-base-patch16
alpha=.1
# dataset path
dst=/sharefs/czz/datasets/multi-clip/cc3m-zh
# dst=/sharefs/czz/datasets/laion28m
bs=256
gpus=8
warmup_steps=1000
kd_type=kd
# run_name is also output path
run_name=rob_large_${gpus}_${loss_fn}_${pooler_fn}_wd${wd}_bs${bs}_lr${lr}_warm${warmup_steps}_ep${ep}_sd${seed}_${kd_type}_enandzh_cc3m


# WANDB_PROJECT=clip-kd HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python /home/chenzhongzhi/multi-clip/multi-clip/run_translation.py  \
WANDB_MODE=offline WANDB_PROJECT=bilingual-kd HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m torch.distributed.launch \
    --nproc_per_node $gpus /home/chenzhongzhi/multi-clip/multi-clip/run_translation.py  \
    --model_name_or_path ${student} \
    --do_train \
    --do_eval \
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
    
