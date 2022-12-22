train_dir=/home/chenzhongzhi/czz/datasets/la13m_para5m_multilingual/laion2b_multi18lg/
train_file=$1
echo "node:${RLAUNCH_REPLICA} processing json file: ${train_file}"

gpus=1
nnodes=1
if [ $gpus -gt 1 ] ;then
    gpus="-m torch.distributed.launch \
    --nproc_per_node=$gpus "
    if [ $nnodes -gt 1 ] ;then
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
run_name=inferece_clip14

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python $gpus run_predict.py  \
    --output_dir /home/chenzhongzhi/ckpt/$run_name \
    --do_pred \
    --max_seq_len 75 \
    --model_name_or_path openai/clip-vit-large-patch14 \
    --test_file  $train_file \
    --image_column source_text \
    --caption_column source_text \
    --remove_unused_columns=False \
    --per_device_eval_batch_size="512" \
    --label_names labels \
    --eval_accumulation_steps 100 \
    --overwrite_output_dir
    
