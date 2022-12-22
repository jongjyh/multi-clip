# exp
lr=1e-4
wd=2e-1
ep=10
seed=42
loss_fn=mse
pooler_fn=cls
layer_kd=false
task=multi-clip
warmup_steps=500
student=xlm-roberta-large
# teacher=laion/CLIP-ViT-H-14-laion2B-s32B-b79K
teacher=openai/clip-vit-large-patch14
bs=128
alpha=.1
kd_type=postkd
run_name=${kd_type}_18lg
# train='none  --train_file /home/chenzhongzhi/czz/datasets/la13m_para5m_multilingual/laion2b_multi18lg/*.json'

# full M19 dataset
# train='/sharefs/baai-mrnd/czz/m18_tokenized'

# dataset for testing and debugging
# train='none  --train_file /sharefs/baai-mrnd/czz/datasets/la13m_para5m_multilingual/multi18m_18lgs_en++.json'
train='none  --train_file /sharefs/baai-mrnd/czz/datasets/la13m_para5m_multilingual/test_1m.json'

# multinode multigpu settings
gpus=1
nnodes=1
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

HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python $gpus \
    run_translation.py  \
    --model_name_or_path ${student} \
    --do_eval \
    --warmup_steps ${warmup_steps} \
    --source_lang zh \
    --target_lang en \
    --max_source_length 75 \
    --num_train_epochs $ep \
    --weight_decay $wd \
    --learning_rate $lr \
    --seed $seed \
    --report_to none \
    --evaluation_strategy steps \
    --run_name ${run_name} \
    --logging_steps 500 \
    --output_dir ckpt/${run_name} \
    --dataset_name ${train} \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 256 \
    --save_strategy epoch \
    --metric_for_best_model eval_cossim_loss \
    --greater_is_better false \
    --loss_fn ${loss_fn} \
    --pooler_fn ${pooler_fn}  \
    --layer_kd ${layer_kd} \
    --teacher_model ${teacher} \
    --alpha ${alpha} \
    --ddp_find_unused_parameters true \
    --prekd_ckpt /home/chenzhongzhi/save_ckpt/xlm1024-33m-cls_ft \
    --kd_type ${kd_type} 