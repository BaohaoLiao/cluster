# copy from https://github.com/facebookresearch/LLM-QAT/blob/main/utils/process_args.py
# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class ModelArguments:
    student_model_path_or_name: Optional[str] = field(
        default="test-input", metadata={"help": "Input model relative manifold path"}
    )
    teacher_model_path_or_name: Optional[str] = field(
        default="test-input", metadata={"help": "Input model relative manifold path"}
    )
    output_model_local_path: str = field(
        default=None, metadata={"help": "Output model local path, do not set manually"}
    )


@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    train_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Train data local path"}
    )
    eval_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Eval data local path"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). 512 or 1024"
        },
    )
    use_kd: Optional[bool] = field(default=False)
    kd_loss_scale: Optional[float] = field(
        default=1.0,
        metadata={"help": "Scale of KD loss."},
    )


def process_args():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    assert model_args.output_model_local_path is not None
    os.makedirs(model_args.output_model_local_path, exist_ok=True)
    return model_args, data_args, training_args