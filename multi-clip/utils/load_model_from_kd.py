
path = '/home/chenzhongzhi/ckpt/multi-clip/baai/selfsupervised_hfl/chinese-roberta-wwm-ext_openai/clip-vit-base-patch32_mse_average_lkdfalse_loss.1_wd1e-4_bs256lr2e-5_ep10_sd42_dst/home/chenzhongzhi/czz/datasets/multi-clip/cc3m-zh/checkpoint-44500'

from modeling_kd import KDmodel
from transformers import PretrainedConfig

config = PretrainedConfig.from_pretrained(path)
model = KDmodel.from_pretrained(path,config=config)


