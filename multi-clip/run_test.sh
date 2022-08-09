lr=2e-5
wd=1e-1
ep=10
seed=42
run_name=chinese-roberta-base_clip-vit-l-14_cosine_cls_wd${wd}_lr${lr}_ep${ep}_sd${seed}
CUDA_VISIBLE_DEVICES=0  python -m debugpy --listen 5678 /home/chenzhongzhi/repo/multi-clip/multi-clip/run_translation.py  \
    --model_name_or_path ckpt/${run_name} \
    --do_predict \
    --source_lang zh \
    --target_lang en \
    --max_source_length 75 \
    --seed $seed \
    --pad_to_max_length true \
    --report_to wandb \
    --run_name ${run_name} \
    --dataset_name /home/chenzhongzhi/repo/multi-clip/multi-clip/cc100k-zh \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=128 \
    --overwrite_output_dir \
