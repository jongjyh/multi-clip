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
from models.modeling_berts import RobertaSeriesConfig
from trainer import OurTrainer
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from transformers import (
    CLIPFeatureExtractor
)
from models.modeling_chclip import CHCLIPProcess, ChCLIPConfig, DoubleCLIPWithKD
import torch.nn as nn
import torch

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.18.0")

require_version("datasets>=1.8.0",
                "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
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

    source_lang: str = field(default=None, metadata={
                             "help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={
                             "help": "Target language id for translation."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines)."})
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
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        elif self.source_lang is None or self.target_lang is None:
            raise ValueError(
                "Need to specify the source language and the target language.")

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

    teacher_model: str = field(default=None, metadata={
                               "help": "to be continued."})
    student_model: str = field(default=None, metadata={
                               "help": "to be continued."})
    variant: str = field(default='invert', metadata={
        "help": "to be continued."})


def get_pretrained_model(kd_config, tokenizer, device, rank):
    # model config
    ###############################
    config = ChCLIPConfig.from_pretrained(kd_config.teacher_model)
    config.text_config = RobertaSeriesConfig.from_pretrained(kd_config.student_model,
                                                             project_dim=config.projection_dim
                                                             )
    config.text_model_name = kd_config.student_model
    config.vision_model_name = kd_config.teacher_model
    config.variant = kd_config.variant
    ###############################

    # model = DoubleCLIPWithKD(config)
    model = DoubleCLIPWithKD(config)
    vision_model = model.vision_model if (rank == 0 or rank==-1) else None
    model.vision_model = None 
    
    teacher_model = AutoModel.from_pretrained(kd_config.teacher_model).text_model
    teacher_model = teacher_model.to(device).eval()
    
    # tokenizer and feature_exactor
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        kd_config.teacher_model)
    teacher_tokenizer = AutoTokenizer.from_pretrained(kd_config.teacher_model)

    processor = CHCLIPProcess(feature_extractor, tokenizer)
    return (model, teacher_model, processor, teacher_tokenizer,vision_model)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, KDArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, kd_args = parser.parse_args_into_dataclasses()

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
    if data_args.dataset_name is not None and data_args.train_file is None:
        # load from local
        raw_datasets = datasets.load_from_disk(data_args.dataset_name)
        # Downloading and loading a dataset from the hub.
        # raw_datasets = load_dataset(
        #     data_args.dataset_name,
        #     cache_dir=model_args.cache_dir,
        #     use_auth_token=True if model_args.use_auth_token else None,
        # )
    else:
        if '*' in data_args.train_file:
            data_files = glob(data_args.train_file)
            raw_train = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets = raw_train['train'].train_test_split(test_size=10000)
            raw_datasets['validation'] = raw_datasets.pop('test')
        else:
            data_files = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
                extension = data_args.train_file.split(".")[-1]
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
                extension = data_args.validation_file.split(".")[-1]
            if data_args.test_file is not None:
                data_files["test"] = data_args.test_file
                extension = data_args.test_file.split(".")[-1]
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
        
    model, teacher_model, processor, teacher_tokenizer, vision_model = get_pretrained_model(
        kd_args, tokenizer, training_args.device,training_args.local_rank)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    source_attr = "target_text"
    target_attr = "source_text"

    def preprocess_function(examples):

        inputs = [ex for ex in examples[source_attr]]
        targets = [ex for ex in examples[target_attr]]
        model_inputs = processor(
            inputs, max_length=data_args.max_source_length, padding=False, truncation=True)
        teacher_inputs = teacher_tokenizer(
            targets, max_length=data_args.max_source_length, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            teacher_features = teacher_model(**teacher_inputs)[1]
            model_inputs['labels'] = teacher_features
        return model_inputs
    # # for our dataset
    # source_attr = "caption_zh"
    # target_attr = "caption"
    # # for multilingual dataset
    # languages = ['zh', 'en'] if 'enzh' in training_args.run_name  else ['zh', 'en','cs','de','fr','ja']
    # def preprocess_function(examples):
    #     # uc2 dataset
    #     if isinstance(examples['caption'][0], list):
    #         targets = [dicts[0]['en'] for dicts in examples['caption']
    #                    for language,caption in dicts[0].items() if language in languages and caption is not None ]
    #         inputs = [dicts[0][language] for dicts in examples['caption']
    #                   for language,caption in dicts[0].items() if language in languages and caption is not None ]
    #     else:
    #         inputs = [ex for ex in examples[source_attr]]
    #         targets = [ex for ex in examples[target_attr]]
    #     model_inputs = processor.tokenizer(
    #         inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
    #     teacher_inputs = teacher_tokenizer(targets, max_length=data_args.max_source_length,
    #                                        padding=True, truncation=True, return_tensors='pt').to(training_args.device)
    #     with torch.no_grad():
    #         teacher_features = teacher_model(**teacher_inputs)[1]
    #         model_inputs['labels'] = teacher_features

    #     return model_inputs
    # new_fingerprint = "{teacher}_{seqlen}_{file}_{languages}{subset}".format(
    #     teacher=kd_args.teacher_model.split('/')[-1],
    #     seqlen=data_args.max_source_length,
    #     file=data_args.train_file.split('/')[-1],
    #     languages=len(languages) if len(languages) == 6 else languages,# for 6 languages
    #     subset="300k" if '300k' in training_args.run_name else "" # for 300k subset
    # )
    # logger.info(
    #     "if you change the teacher or data please check the fingerprint.")
    # logger.info(f"now fingerprint for train:{new_fingerprint}")

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=1,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
                # new_fingerprint=new_fingerprint
            )

    # new_fingerprint = "{teacher}_{seqlen}_{file}_{languages}{subset}".format(
    #     teacher=kd_args.teacher_model.split('/')[-1],
    #     seqlen=data_args.max_source_length,
    #     file=data_args.train_file.split('/')[-1],
    #     languages=len(languages) if len(languages) == 6 else languages,# for 6 languages
    #     subset="300k" if '300k' in training_args.run_name else ""
    # )
    # logger.info(f"now fingerprint for eval:{new_fingerprint}")
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
                # new_fingerprint=new_fingerprint
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(
                range(max_predict_samples))
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
    label_pad_token_id = - \
        100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = None

    def compute_metrics(eval_preds):
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics import mean_squared_error
        preds, labels = eval_preds
        clip_outputs, teacher_outputs = preds, labels
        if isinstance(preds, tuple):
            # clip_outputs,direct_outputs,merge_outputs,teacher_outputs = preds
            clip_outputs, teacher_outputs = preds

        # cossim_loss_fn = torch.nn.CosineEmbeddingLoss()
        # cossim_loss = cossim_loss_fn(torch.from_numpy(direct_outputs),torch.from_numpy(teacher_outputs),torch.Tensor([1.])).item()
        # di_cos_loss = np.diag(cosine_similarity(direct_outputs,teacher_outputs)).mean()
        # di_mse_loss = mean_squared_error(teacher_outputs,direct_outputs)
        # mg_cos_loss = np.diag(cosine_similarity(merge_outputs,teacher_outputs)).mean()
        # mg_mse_loss = mean_squared_error(teacher_outputs,merge_outputs)
        cp_cos_loss = np.diag(cosine_similarity(
            clip_outputs, teacher_outputs)).mean()
        cp_mse_loss = mean_squared_error(teacher_outputs, clip_outputs)

        result = {
            # "di_cossim":di_cos_loss,
            # "di_mse":        di_mse_loss,
            # "mg_cossim":mg_cos_loss,
            # "mg_mse":        mg_mse_loss,
            # "cp_cossim":     cp_cos_loss,
            # "cp_mse":        cp_mse_loss,
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
        compute_metrics=compute_metrics,
    )
    trainer.processor = processor
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
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        if trainer.is_world_process_zero():
            logger.info("*** Evaluate ***")
            dataset_attr = (
                "flickr30k", "flickr30k-cn", "imagenet1k", "imagenet1k_zh"
                # "flickr30k", "flickr30k-cn"
            )
            from CLIP_benchmark_internal.evaluate import evaluate
            dataset_metrics = {}
            for name in dataset_attr:
                metrics = evaluate(
                    dataset_name=name,
                    model_name=training_args.output_dir,
                    pretrained='pretrained',
                    output=f'/home/chenzhongzhi/results/{training_args.run_name}.json',
                    dataset_root="/sharefs/baai-mmdataset/clip_benchmark_datasets",
                    recall_k=[1, 5, 10],
                    model=model,
                    processor=processor
                )

                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

                for key in metrics.keys():
                    dataset_metrics[f"eval_{name}_{key}"] = metrics[key]
            trainer.log(dataset_metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(
                predict_dataset)
        )
        metrics["predict_samples"] = min(
            max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(
                    training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path,
              "tasks": "translation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()
