cd /sharefs/czz/repo/Chinese-CLIP
split=valid # 指定计算valid或test集特征
dataset=muge
# model=bs-bs-3m
# model=bs-bs-8m
# model=lg-lg-3m
# model=lg-lg-8m
model=lg30M

# for Chinese-CLIP
# path='/sharefs/czz/repo/pretrained_weights/clip_cn_vit-b-16.pt'

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
echo ''
echo 'T2I'
cat T2Ioutput.json
echo 'I2T'
cat I2Toutput.json