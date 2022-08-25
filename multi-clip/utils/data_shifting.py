import torch.nn as nn
import torch
import torch.nn.functional as F
import regex as re



def en_word_embedding_zero_shifting(model,tok,eps=1e-2):
    # make sure model is a text encoder
    # assert isinstance(model,)
    emb=model.get_input_embeddings()
    vocab = tok.get_vocab()
    en_ids,zh_ids = [],[]
    en_dict,zh_dict = {},{}
    for k,v in vocab.items() :
        if re.search(pattern=r"[a-zA-Z]",string=k):
            en_ids.append(v)
            en_dict[k]=v
        elif re.search(pattern=r"[\u4e00-\u9fa5]",string=k):
            zh_ids.append(v)
            zh_dict[k]=v
    # en_embeds,zh_embeds = emb.weight.data[torch.tensor(en_ids)],emb.weight.data[torch.tensor(zh_ids)]
    # zero shifting
    emb.weight.data[torch.tensor(en_ids)]= emb.weight.data[torch.tensor(en_ids)].normal_(0,eps)

def en_word_embedding_zero_norm(model,tok,):
    # make sure model is a text encoder
    # assert isinstance(model,)
    emb=model.get_input_embeddings()
    vocab = tok.get_vocab()
    en_ids,zh_ids = [],[]
    en_dict,zh_dict = {},{}
    for k,v in vocab.items() :
        if re.search(pattern=r"[a-zA-Z]",string=k):
            en_ids.append(v)
            en_dict[k]=v
        elif re.search(pattern=r"[\u4e00-\u9fa5]",string=k):
            zh_ids.append(v)
            zh_dict[k]=v
    # en_embeds,zh_embeds = emb.weight.data[torch.tensor(en_ids)],emb.weight.data[torch.tensor(zh_ids)]
    # zero shifting

    emb.weight.data[torch.tensor(en_ids)]= emb.weight.data[torch.tensor(en_ids)] - torch.randn((768)).normal_(0,1e-1)

def zh_en_word_embedding_shift(model,tok,):
    # make sure model is a text encoder
    # assert isinstance(model,)
    emb=model.get_input_embeddings()
    vocab = tok.get_vocab()
    en_ids,zh_ids = [],[]
    en_dict,zh_dict = {},{}
    for k,v in vocab.items() :
        if re.search(pattern=r"[a-zA-Z]",string=k):
            en_ids.append(v)
            en_dict[k]=v
        elif re.search(pattern=r"[\u4e00-\u9fa5]",string=k):
            zh_ids.append(v)
            zh_dict[k]=v
    en_embeds,zh_embeds = emb.weight.data[torch.tensor(en_ids)],emb.weight.data[torch.tensor(zh_ids)]
    mean_en,mean_zh = en_embeds.mean(dim=0),zh_embeds.mean(dim=0)
    dis = mean_en - mean_zh
    # shifting
    _zh_embeds = zh_embeds 
    emb.weight.data[torch.tensor(zh_ids)]= _zh_embeds + dis
    _en_embeds = en_embeds
    emb.weight.data[torch.tensor(en_ids)]= _en_embeds + dis