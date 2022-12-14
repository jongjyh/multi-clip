#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from glob import glob
import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Trainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from transformers import (
    CLIPProcessor,
)
from open_source_models.modeling_altclip import AltCLIPConfig,AltCLIP
from open_source_models.processing_altclip import AltCLIPProcessor
from open_source_models.modeling_kd import KDmodel
import torch.nn as nn
from typing import Mapping
import torch
from kd_trainer import OurTrainer
from transformers.data.data_collator import DataCollatorForLanguageModeling,_torch_collate_batch
from clip.clip import tokenize 
from clip.hf_model import HFTextEncoder
# from utils.prekd_adapter import adding_adapter_layer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.18.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)

    
# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]


class OurDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            examples_ = [ {
                            'input_ids':i.pop('teacher_input_ids'),
                            'attention_mask':i.pop('teacher_attention_mask')
                        } for i in examples]

            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
            batch_ = self.tokenizer.pad(examples_, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
            batch.data['teacher_input_ids']=batch_.pop('input_ids')
            batch.data['teacher_attention_mask']=batch_.pop( 'attention_mask' )
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["mlm_input_ids"], batch["mlm_labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            pass
            # labels = batch["input_ids"].clone()
            # if self.tokenizer.pad_token_id is not None:
            #     labels[labels == self.tokenizer.pad_token_id] = -100
            # batch["labels"] = labels
        return batch

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    sub_train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on "
            "a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (sacreblue) on " "a jsonlines file."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
            "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
            "needs to be the target language token.(Usually it is the target language token)"
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        elif self.source_lang is None or self.target_lang is None:
            raise ValueError("Need to specify the source language and the target language.")

        # accepting both json and jsonl file extensions, as
        # many jsonlines files actually have a .json extension
        valid_extensions = ["json", "jsonl"]

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in valid_extensions, "`train_file` should be a jsonlines file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in valid_extensions, "`validation_file` should be a jsonlines file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

@dataclass
class KDArguments:
    """
    to be continued.
    """

    loss_fn: str = field(default=None, metadata={"help": "to be continued."})
    pooler_fn: str = field(default=None, metadata={"help": "to be continued."})
    layer_kd: bool = field(default=False, metadata={"help": "to be continued."})
    teacher_model: str = field(default=None, metadata={"help": "to be continued."})
    alpha: float = field(default=.1, metadata={"help": "to be continued."})
    kd_type: str = field(default='kd', metadata={"help": "to be continued."})
    prekd_ckpt: str = field(default=None, metadata={"help": "to be continued."})
    delta: str = field(default=None, metadata={"help": "to be continued."})

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments,KDArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args,kd_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with "
            "`--source_prefix 'translate English to German: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None and data_args.dataset_name != 'none':
        # load from local 
        raw_datasets = datasets.load_from_disk(data_args.dataset_name)
        datasets_ = raw_datasets.train_test_split(test_size=5000)
        datasets_['validation'] = datasets_.pop('test')
        raw_datasets = datasets_
    else:
        # data_files = {}
        data_files = glob(data_args.train_file)
        # if data_args.train_file is not None:
        #     data_files["train"] = data_args.train_file
        #     extension = data_args.train_file.split(".")[-1]
        # if data_args.validation_file is not None:
        #     data_files["validation"] = data_args.validation_file
        #     extension = data_args.validation_file.split(".")[-1]
        # if data_args.test_file is not None:
        #     data_files["test"] = data_args.test_file
        #     extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        datasets_ = raw_datasets['train'].train_test_split(test_size=5000) 
        datasets_['validation'] = datasets_.pop('test')
        raw_datasets = datasets_
        
        if data_args.sub_train_file is not None:
            sub_dataset = load_dataset(
                'json',
                data_files=data_args.sub_train_file,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )           
            sub_dataset = sub_dataset['train']
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    eva=False 
    mlm=True
    from transformers import PretrainedConfig
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    processor = CLIPProcessor.from_pretrained(kd_args.teacher_model)
    mix_processor = AltCLIPProcessor(feature_extractor=processor.feature_extractor,tokenizer=tokenizer)
    kd_config_dict = {
        "teacher_model":kd_args.teacher_model,
        "student_model":model_args.model_name_or_path,
        "loss_fn":kd_args.loss_fn,
        "pooler_fn":kd_args.pooler_fn,
        "layer_kd":kd_args.layer_kd,
        "alpha":kd_args.alpha,
        "learn_encoder":False,
        "kd_type":kd_args.kd_type,
        "add_lm_task":mlm,
        "eva":eva,
    }
    kd_config = PretrainedConfig(**kd_config_dict)
    if kd_args.kd_type == 'kd':
    # for origin kd
        model = KDmodel(kd_config,EVA=eva) 
    elif kd_args.kd_type == 'postkd':
    # for post KD
        model = KDmodel(kd_config) 
        pre_clip = AltCLIP.from_pretrained(kd_args.prekd_ckpt)
        model.student = pre_clip.text_model
        model.student_config = model.student.config
        model.student.add_lm_task = True
    elif 'prekd' in kd_args.kd_type :
    # for pre kd
        model = KDmodel(kd_config)
        # use adapter
        if kd_args.delta == 'adapter' :
            adding_adapter_layer(model.student,model.student.config.project_dim)
                
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert data_args.target_lang is not None and data_args.source_lang is not None, (
            f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --source_lang and "
            "--target_lang arguments."
        )

        tokenizer.src_lang = data_args.source_lang
        tokenizer.tgt_lang = data_args.target_lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Temporarily set max_target_length for training.
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )
        
    source_attr = "caption_zh"
    # # source_attr = "caption"
    target_attr = "caption"
    def preprocess_function_train(examples):
        inputs = [ex for ex in examples[source_attr]]
        targets = [ex for ex in examples[target_attr]]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        eng_inputs = tokenizer(targets, max_length=data_args.max_source_length, padding=padding, truncation=True) 
        for (k,v),(k1,v1) in zip(model_inputs.items(),eng_inputs.items()):
            model_inputs[k1] = [*v,*v1]

        teacher_inputs = processor(targets, max_length=data_args.max_source_length,padding=padding,truncation=True,)

        model_inputs['teacher_input_ids'] = [ *teacher_inputs['input_ids'],*teacher_inputs['input_ids'] ] 
        model_inputs['teacher_attention_mask'] = [ *teacher_inputs['attention_mask'],*teacher_inputs['attention_mask'] ]

        return model_inputs

    # source_attr = "caption_zh"
    source_attr = "target_text"
    target_attr = "source_text"
    def preprocess_function(examples):

        inputs = [ex for ex in examples[source_attr]]
        targets = [ex for ex in examples[target_attr]]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        teacher_inputs = processor(targets, max_length=data_args.max_source_length,padding=padding,truncation=True,)
        model_inputs['teacher_input_ids'] = teacher_inputs['input_ids'] 
        model_inputs['teacher_attention_mask'] = teacher_inputs['attention_mask']
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
            if data_args.sub_train_file is not None:
                sub_dataset = sub_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                batch_size=10000,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="running tokenizer on train dataset",
            )
            # train_dataset.save_to_disk("/sharefs/baai-mrnd/czz/m18_tokenized")
            # print("save done.")
        if data_args.sub_train_file is not None:
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                sub_dataset = sub_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="running tokenizer on train dataset",
                )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=None,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = OurDataCollatorForLanguageModeling(
            tokenizer,
            mlm_probability=0.15,
        )
        data_collator.mlm = mlm

    # Metric
    # metric = load_metric("sacrebleu")

    # def postprocess_text(preds, labels):
    #     preds = [pred.strip() for pred in preds]
    #     labels = [[label.strip()] for label in labels]

    #     return preds, labels

    # def compute_metrics(eval_preds):
    #     preds, labels = eval_preds
    #     if isinstance(preds, tuple):
    #         preds = preds[0]
    #     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #     if data_args.ignore_pad_token_for_loss:
    #         # Replace -100 in the labels as we can't decode them.
    #         labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    #     # Some simple post-processing
    #     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    #     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    #     result = {"bleu": result["score"]}

    #     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    #     result["gen_len"] = np.mean(prediction_lens)
    #     result = {k: round(v, 4) for k, v in result.items()}
    #     return result

    def compute_metrics(eval_preds):
        preds, _ = eval_preds
        if isinstance(preds, tuple):
            if mlm:
                x,y,mse_loss,mlm_loss = preds
            else :
                _,x,y = preds
        
        # convert to tensor 
        x,y = torch.from_numpy(x),torch.from_numpy(y)

        cosine_sim = nn.CosineEmbeddingLoss()
        mse = nn.MSELoss()
        l1 = nn.L1Loss()
        assert len(x) == len(y)
        cos_loss = cosine_sim(x,y,torch.Tensor([1.])).item()
        mse_loss = mse(x,y).item() 
        l1_loss = l1(x,y).item()
        mlm_loss = mlm_loss.mean() if mlm else 0.
        
        result = {
            "cossim_loss":cos_loss,
            "mse":mse_loss,
            "l1_loss":l1_loss,
            "mlm_loss":mlm_loss
        }

        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = OurTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics ,
    )
    trainer.do_mlm = mlm
    if data_args.sub_train_file is not None:
        trainer.sub_dataset = sub_dataset
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate( metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))

    # save as clip model
    if training_args.local_rank==0 or training_args.local_rank == -1 :
        if eva:
            hf_encoder = HFTextEncoder(
                "xlm-roberta-large",
                output_dim=model.text_projection.data.shape[0],
                config=model.student_config,
                proj='linear',
            )
            hf_encoder.transformer = model.student
            hf_encoder.proj = model.text_projection
            model.student_config.save_pretrained(training_args.output_dir)
            mix_processor.tokenizer.save_pretrained(training_args.output_dir)
            torch.save(hf_encoder.state_dict(),training_args.output_dir + "/hf_encoder.pt")
        else:
            kd_model = trainer.model
            chinese_clip_config = AltCLIPConfig.from_pretrained(kd_config.teacher_model)
            chinese_clip_config.text_config = kd_model.student_config
            chinese_clip = AltCLIP.from_pretrained(kd_config.teacher_model,ignore_mismatched_sizes=True,config=chinese_clip_config)
            chinese_clip.text_model = kd_model.student
            chinese_clip.save_pretrained(training_args.output_dir)
            mix_processor.save_pretrained(training_args.output_dir)



    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "translation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    languages = [l for l in [data_args.source_lang, data_args.target_lang] if l is not None]
    if len(languages) > 0:
        kwargs["language"] = languages

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results




if __name__ == "__main__":
    main()
