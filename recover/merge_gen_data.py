# Copy from https://github.com/facebookresearch/LLM-QAT/blob/main/merge_gen_data.py
# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import fire
from tqdm import tqdm

def main(save_dir: str, num_shards: int):
    all_text = []

    for shard_idx in tqdm(range(num_shards)):
        for line in open(f"{save_dir}/gen.chunk.{str(shard_idx).zfill(2)}-{str(num_shards).zfill(2)}.jsonl", 'r'):
            all_text.append(json.loads(line))

    with open(f"{save_dir}all_gen.jsonl", "a") as f:
        for i in range(len(all_text)):
            f.write(json.dumps(all_text[i]))
            f.write('\n')

if __name__ == "__main__":
    fire.Fire(main)