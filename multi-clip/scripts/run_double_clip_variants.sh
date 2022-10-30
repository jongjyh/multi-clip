export PYTHONPATH=$PYTHONPATH:/home/chenzhongzhi/multi-clip/multi-clip/CLIP_benchmark_internal
# exp
lr=1e-4
wd=1e-1
ep=10
seed=42
task=multi-clip
student=xlm-roberta-base
teacher=openai/clip-vit-base-patch16
bs=180
warmup_steps=1000

# direct is kd baseline, invert is bart-alike.
# variant=direct
variant=invert

# dataset setting
uc2=1
if [ $uc2 -eq 0 ] ;then
    dst="/sharefs/czz/datasets/multi-clip/cc3m-zh"
else
    train="/sharefs/baai-mrnd/czz/datasets/cc3m_uc2/train_cc3m.json"
    eval="/sharefs/baai-mrnd/czz/datasets/cc3m_uc2/eval_cc3m.json"
    dst="none --train_file ${train} --validation_file ${eval}"
fi

# multi gpu setting
gpus=4
if [ $gpus -gt 1 ] ;then
    gpus="-m torch.distributed.launch --nproc_per_node $gpus"
else
    gpus=""
fi

# be careful to set languages, cause we use languages to hash dataset.
# languages=enzh
# languages=6lgs
languages=6lgs_300k
if echo $languages | grep -q "300k"; then
    exp="${exp} --max_train_samples 300000"
    echo "using 300k subset"
else
    echo "using full dataset"
fi
run_name=${variant}_cc3muc2_xlmBase_basep16_bs${bs}_wd${wd}_lr${lr}_ep${ep}_ws${warmup_steps}_doubleclip_$languages

# debug setting
debug=0
if [ $debug -eq 1 ] ;then
    debug="
    --max_train_samples 9000 \
    --max_eval_samples 1000 --overwrite_output_dir --warmup_steps 0 --logging_steps 100
    "
    run_name=${run_name}_debug
    gpus="-m debugpy --listen 5679"
    # gpus=""
else
    debug=""
fi


WANDB_MODE=offline WANDB_PROJECT=double-clip HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python ${gpus} \
    /home/chenzhongzhi/multi-clip/multi-clip/run_kd.py  \
    --model_name_or_path ${student} \
    --do_train \
    --do_eval \
    --warmup_steps ${warmup_steps} \
    --source_lang zh \
    --target_lang en \
    --max_source_length 75 \
    --max_eval_samples 10000 \
    --num_train_epochs $ep \
    --remove_unused_columns false \
    --weight_decay $wd \
    --learning_rate $lr \
    --seed $seed \
    --report_to wandb \
    --evaluation_strategy steps \
    --save_total_limit 1 \
    --run_name ${run_name} \
    --logging_steps 500 \
    --save_steps 2000 \
    --output_dir ckpt/${run_name} \
    --dataset_name $dst \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 256 \
    --metric_for_best_model eval_flickr30k-cn_mean_retrieval_recall \
    --greater_is_better 1 \
    --teacher_model ${teacher} \
    --student_model ${student} \
    --variant $variant $exp $debug 