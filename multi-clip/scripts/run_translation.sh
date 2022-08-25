lr=2e-5
wd=1e-4
ep=1
seed=42
loss_fn=mse
pooler_fn=average
layer_kd=true
task=multi-clip
student=hfl/chinese-roberta-wwm-ext
teacher=openai/clip-vit-large-patch14
run_name=${student}_${teacher}_${loss_fn}_${pooler_fn}_lkd${layer_kd}_nolinear_wd${wd}_lr${lr}_ep${ep}_sd${seed}
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m debugpy --listen 5678 /home/chenzhongzhi/repo/multi-clip/multi-clip/run_translation.py  \
    --model_name_or_path ${student} \
    --do_train \
    --do_eval \
    --source_lang zh \
    --target_lang en \
    --max_source_length 75 \
    --num_train_epochs $ep \
    --weight_decay $wd \
    --learning_rate $lr \
    --seed $seed \
    --pad_to_max_length true \
    --report_to none \
    --evaluation_strategy steps \
    --save_total_limit 1 \
    --run_name ${run_name} \
    --logging_steps 500 \
    --output_dir ckpt/${task}/${run_name} \
    --dataset_name /home/chenzhongzhi/czz/datasets/multi-clip/cc100k-zh \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --loss_fn ${loss_fn} \
    --pooler_fn ${pooler_fn}  \
    --layer_kd ${layer_kd} \
    --overwrite_output_dir \
    --teacher_model ${teacher} \
    --load_best_model_at_end \
    --max_train_samples 128 \
    --max_eval_samples 128  
