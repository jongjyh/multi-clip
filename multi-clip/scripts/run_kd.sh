# exp
lr=1e-4
wd=2e-1
ep=20
seed=42
loss_fn=mse
pooler_fn=cls
layer_kd=false
task=multi-clip
student=xlm-roberta-large
teacher=openai/clip-vit-large-patch14
# teacher=openai/clip-vit-base-patch16
alpha=.1
# dataset path
train=/sharefs/baai-mrnd/czz/datasets/la28m-cc3m-ts5m/train.json
eval=/sharefs/baai-mrnd/czz/datasets/la28m-cc3m-ts5m/eval.json
bs=128

# multinode multigpu settings
gpus=8
nnodes=6
if [ $gpus -gt 1 ] ;then
    gpus="-m torch.distributed.launch \
    --nproc_per_node=$gpus "
    if [ $nnodes -gt 1 ] ;then
        source multinode.sh
        port=29502
        masterip=$masterip
        gpus="${gpus} --nnodes=$nnodes \
        --node_rank=$RLAUNCH_REPLICA \
        --master_addr=$masterip \
        --master_port=$port"
    fi
else
    gpus=""
fi
warmup_steps=500
kd_type=kd
# run_name is also output path
run_name=xlm_large_${loss_fn}_${pooler_fn}_wd${wd}_bs${bs}_lr${lr}_warm${warmup_steps}_ep${ep}_sd${seed}_${kd_type}_enandzh_cc3m


# WANDB_PROJECT=clip-kd HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python /home/chenzhongzhi/multi-clip/multi-clip/run_translation.py  \
WANDB_MODE=offline WANDB_PROJECT=bilingual-kd HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python $gpus \
    /home/chenzhongzhi/multi-clip/multi-clip/run_translation.py  \
    --model_name_or_path ${student} \
    --do_train \
    --do_eval \
    --warmup_steps ${warmup_steps} \
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
    --output_dir ckpt/${run_name} \
    --train_file ${train}  --validation_file ${eval} \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 512 \
    --save_steps 2000 \
    --metric_for_best_model eval_cossim_loss \
    --greater_is_better false \
    --loss_fn ${loss_fn} \
    --pooler_fn ${pooler_fn}  \
    --layer_kd ${layer_kd} \
    --teacher_model ${teacher} \
    --load_best_model_at_end \
    --alpha ${alpha} \
    --kd_type ${kd_type} 
    