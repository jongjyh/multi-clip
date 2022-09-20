<div align="center">    
 
# Multilingual CLIP

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->


<!--  
Conference   
-->   
</div>
 
## Description   
This is a repo for training a multiligual CLIP by knowledge distilation.

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/multi_clip

# install project   
cd multi_clip
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   

## training
Training pipeline is done by `transformers`. All available training scripts all located in `multi-clip/scripts`.
Before training, you should specified 
- dataset path( which could be loaded by `datasets` library )
- teacher and student backbone model (e.g xlm-r, bert )
- output path
```bash
# exp
lr=1e-4
wd=1e-4
ep=10
seed=42
loss_fn=cl
pooler_fn=cls
layer_kd=false
task=multi-clip
student=xlm-roberta-base
# student=hfl/chinese-roberta-wwm-ext
# teacher=openai/clip-vit-large-patch14
teacher=openai/clip-vit-base-patch16
alpha=.1
# dataset path
dst=/sharefs/czz/datasets/multi-clip/cc3m-zh
# dst=/sharefs/czz/datasets/laion28m
bs=512
gpus=2
warmup_steps=1000
kd_type=kd
# run_name is also output path
run_name=xlm_base_${gpus}_${loss_fn}_${pooler_fn}_wd${wd}_bs${bs}_lr${lr}_warm${warmup_steps}_ep${ep}_sd${seed}_${kd_type}
```
After setting that, you could run training script by:

```bash
cd multi-clip/multi-clip
bash scripts/run_kd.sh
```
