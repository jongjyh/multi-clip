# %%
from modeling_chclip import *
from transformers import CLIPModel
path = '/home/chenzhongzhi/ckpt/multi-clip/baai/hfl/chinese-roberta-wwm-ext-large_openai/clip-vit-large-patch14_mse_average_lkdfalse_loss.1_wd1e-4_lr2e-5_ep10_sd42_dst/home/chenzhongzhi/czz/datasets/multi-clip/cc100k-zh/student_model'

# wrong dim in text_projection
zh_clip = ChineseCLIP.from_pretrained(path,ignore_mismatched_sizes=True)
zh_clip.config.text_embed_dim = zh_clip.config.text_config.project_dim 
ori_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
zh_clip.text_projection = ori_clip.text_projection

# %%
from transformers import CLIPProcessor,AutoTokenizer
import requests
from PIL import Image

class CHCLIPProcess(CLIPProcessor):
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",tokenizer=tokenizer)
mix_processor = CHCLIPProcess(feature_extractor=processor.feature_extractor,tokenizer=tokenizer)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = mix_processor(text=["一张猫的照片", "一张狗的照片"], images=image, return_tensors="pt", padding=True)

outputs = zh_clip(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

# %%
probs

# %%
mix_processor.save_pretrained(path)
zh_clip.save_pretrained(path)


