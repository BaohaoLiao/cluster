# Copy from https://github.com/facebookresearch/LLM-QAT/blob/main/train.py
# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 2023.07.05 - Add quantization and knowledge distillialtion
#              Meta Platforms, Inc. <zechunliu@meta.com>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import math
import torch
from torch import distributed as dist
import transformers
from transformers import default_data_collator, Trainer

from recover.utils.process_args import process_args
from recover.utils import utils, datautils
from recover.utils.kd_trainer import KDTrainer

from models.llama_layer import CustomLlamaForCausalLM

log = utils.get_logger("clm")

def train():
    dist.init_process_group(backend="nccl")
    model_args, data_args, training_args = process_args()

    log.info("Start to load model...")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    model = CustomLlamaForCausalLM.from_pretrained(
        model_args.student_model_path_or_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        #device_map=None if len(training_args.fsdp) > 0 else "auto",
    )
    model.cuda()

    if training_args.use_kd:
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.teacher_model_path_or_name,
            cache_dir=training_args.cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            #device_map=None if len(training_args.fsdp) > 0 else "auto",
        )
        teacher_model.eval()
        teacher_model.cuda()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.use_cache = False
        model.kd_loss_scale = training_args.kd_loss_scale
        model.teacher = teacher_model
    log.info("Complete model loading...")

    log.info("Start to load tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.student_model_path_or_name,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    log.info("Complete tokenizer loading...")

    train_dataset, valid_dataset = datautils.get_train_val_dataset(
        train_path=data_args.train_data_local_path,
        valid_path=data_args.eval_data_local_path
        if data_args.eval_data_local_path is not None
        else None,
    )
    train_data = datautils.CustomJsonDataset(
        train_dataset, tokenizer, block_size=training_args.model_max_length
    )
    valid_data = None
    if valid_dataset is not None:
        valid_data = datautils.CustomJsonDataset(
            valid_dataset, tokenizer, block_size=min(training_args.model_max_length, 1024)
        )
    model.config.use_cache = False
    if training_args.use_kd:
        myTrainer = KDTrainer
    else:
        myTrainer = Trainer
    trainer = myTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=valid_data if training_args.do_eval else None,
        data_collator=default_data_collator,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_state()
        utils.safe_save_model_for_hf_trainer(trainer, model_args.output_model_local_path)

    # Evaluation
    if training_args.do_eval:
        model.to("cuda")
        metrics = trainer.evaluate()
        max_eval_samples = len(valid_data)
        metrics["eval_samples"] = min(max_eval_samples, len(valid_data))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    torch.distributed.barrier()


if __name__ == "__main__":
    train()