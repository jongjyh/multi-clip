cd /sharefs/czz/repo/Chinese-CLIP
split=test # 指定计算valid或test集特征
dataset=flickr30k-cn
model=bs-bs-3m
# model=bs-bs-8m
# model=lg-lg-3m
# model=lg-lg-3m-enc
# model=lg-lg-8m
# model=xlm768-en-zh-3m
# model=xlm768-freeze-zh-3m
# model=xlm768-en-zh-3m_1e-4
# model=nopretrain_en
# model=pretrain
# model=pretrain_freeze
# model=xlm768-freeze-en-3m
# model=xlm768-en-3m
# model=xlm768-freeze-en-zh
# model=xlm768--en-zh-cls
# model=xlm768-pretrain-en-freeze-cls
# model=xlm768-zh-cls
# model=brivl
# model=kd_3e-4
for model in xlm768-la28-enandzh-cls
do
    path=/sharefs/czz/save_ckpt/${model}
    

    if [ $model != 'brivl' ];then
        bash src/eval/feature_ext.sh $path $dataset $split
    else
        python ~/brivl-nmi/example.py  \
            --dataset ${dataset} \
            --split ${split}
    fi

    bash src/eval/knn.sh $dataset $split
    bash src/eval/T2Irecall.sh $dataset $split

    bash src/eval/i2tpre.sh $dataset $split
    bash src/eval/I2Trecall.sh $dataset $split

    # result
    mv T2Ioutput.json ${model}_T2Ioutput.json 
    mv I2Toutput.json ${model}_I2Toutput.json 
done