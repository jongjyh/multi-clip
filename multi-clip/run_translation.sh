lr=2e-4
wd=1e-1
ep=10
seed=42
run_name=lr_wd_ep_sd
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=1 python -m debugpy --listen 5678 run_translation.py  \
    --model_name_or_path xlm-roberta-large \
    --do_train \
    --source_lang af \
    --target_lang en \
    --max_source_length 75 \
    --pad_to_max_length true \
    --report_to none \
    --run_name ${run_name} \
    --dataset_name M-CLIP/ImageCaptions-7M-Translations \
    --output_dir ckpt/${run_name} \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --overwrite_output_dir \
