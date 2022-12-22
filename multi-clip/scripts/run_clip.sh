# ladder
/home/chenzhongzhi/clash -d /home/chenzhongzhi/clash  &>/home/chenzhongzhi/clash/network.log &
sleep 1s
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=sock5://127.0.0.1:7891

WANDB_PROJECT=clip-cts HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run_clip.py \
    --output_dir /sharefs/czz/
    --model_name_or_path /sharefs/czz/save_ckpt/xlm768-
    --data_dir ./data \
    --train_file   \
    --image_column source_text \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train  --do_eval \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --push_to_hub
