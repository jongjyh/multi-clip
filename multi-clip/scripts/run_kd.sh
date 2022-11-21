# exp
lr=5e-5
wd=2e-1
ep=10
seed=42
loss_fn=mse
pooler_fn=cls
layer_kd=false
task=multi-clip
student=xlm-roberta-large
# teacher=laion/CLIP-ViT-H-14-laion2B-s32B-b79K
teacher=openai/clip-vit-large-patch14
alpha=.1
# dataset path
# train="/home/chenzhongzhi/czz/datasets/la13m_para5m_multilingual/multi18m_12lgs.json  --sub_train_file /home/chenzhongzhi/czz/datasets/la13m_para5m_multilingual/multi18m_9lgs.json"
train="/home/chenzhongzhi/czz/datasets/la13m_para5m_multilingual/multi18m_12lgs.json"

# eval=/sharefs/baai-mrnd/czz/datasets/la28m-cc3m-ts5m/eval.json
bs=32

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
    gpus="-m debugpy --listen 5678"
fi
warmup_steps=500
# kd_type=postkd
kd_type=kd
# run_name is also output path
run_name=${kd_type}_12lgs_18m_lmloss_fs


# WANDB_PROJECT=clip-kd HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python /home/chenzhongzhi/multi-clip/multi-clip/run_translation.py  \
WANDB_MODE=offline WANDB_PROJECT=bilingual-kd HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python $gpus \
    /home/chenzhongzhi/multi-clip/multi-clip/run_translation.py  \
    --model_name_or_path ${student} \
    --do_train \
    --do_eval \
    --warmup_steps ${warmup_steps} \
    --source_lang zh \
    --target_lang en \
    --max_source_length 70 \
    --num_train_epochs $ep \
    --weight_decay $wd \
    --learning_rate $lr \
    --seed $seed \
    --report_to wandb \
    --evaluation_strategy steps \
    --run_name ${run_name} \
    --logging_steps 500 \
    --output_dir ckpt/${run_name} \
    --train_file ${train}  \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 64 \
    --save_strategy epoch \
    --metric_for_best_model eval_cossim_loss \
    --greater_is_better false \
    --loss_fn ${loss_fn} \
    --pooler_fn ${pooler_fn}  \
    --layer_kd ${layer_kd} \
    --teacher_model ${teacher} \
    --alpha ${alpha} \
    --kd_type ${kd_type} \