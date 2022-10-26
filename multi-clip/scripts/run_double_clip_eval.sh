# exp
lr=2e-4
wd=1e-1
ep=10
seed=42
loss_fn=mse
task=multi-clip
student=xlm-roberta-base
teacher=openai/clip-vit-large-patch14
dst=/sharefs/czz/datasets/multi-clip/cc3m-zh
# dst=/sharefs/czz/datasets/laion28m
bs=256
warmup_steps=1000
variant=onlyiv
baseline=false

# multi gpu setting
gpus=1
if [ $gpus -gt 1 ] ;then
gpus="-m torch.distributed.launch --nproc_per_node $gpus"
else
gpus=""
fi

run_name=${variant}_cc3m_xlmBase_p14_bs${bs}_wd${wd}_lr${lr}_ep${ep}_ws${warmup_steps}_doubleclip
# debug setting
debug=0
if [ $debug -eq 1 ] ;then
debug="--max_train_samples 5000"
run_name=${run_name}_debug
debug_py="-m debugpy --listen 5678"
else
debug=""
fi

WANDB_MODE=offline WANDB_PROJECT=double-clip HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python $debug_py ${gpus} \
    /home/chenzhongzhi/multi-clip/multi-clip/run_kd.py  \
    --model_name_or_path onlyiv_cc3m_xlmBase_p14_bs256_wd1e-1_lr2e-4_ep10_ws1000_doubleclip \
    --do_eval \
    --warmup_steps ${warmup_steps} \
    --source_lang zh \
    --target_lang en \
    --max_source_length 40 \
    --num_train_epochs $ep \
    --remove_unused_columns false \
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
    --per_device_eval_batch_size 256 \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --loss_fn ${loss_fn} \
    --teacher_model ${teacher} \
    --baseline ${baseline} \
    $debug