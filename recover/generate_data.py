import os
import json
import fire
import logging
from tqdm import tqdm
import torch
import transformers


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def main(model_name_or_path, save_dir, shard_idx, num_shards, device="cuda:0"):
    logging.info(f"{'-'*20} Loading model and tokenizer {'-'*20}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    n_vocab = model.config.vocab_size // num_shards #each index in the vocab will be used as the first token
    logging.info(f"Generate {n_vocab} samples for {shard_idx}/{num_shards} shard")

    if os.path.exists(f"{save_dir}/gen.chunk.{str(shard_idx).zfill(2)}-{str(num_shards).zfill(2)}.jsonl"):
        with open(f"{save_dir}/gen.chunk.{str(shard_idx).zfill(2)}-{str(num_shards).zfill(2)}.jsonl", "r") as f:
            lines = f.readlines()
            inner_loop = len(lines) % n_vocab
            outer_loop = len(lines) // n_vocab
    else:
        inner_loop = 0
        outer_loop = 0

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for j in range(3 + outer_loop, 4):
        for i in tqdm(range(int(shard_idx) * n_vocab + inner_loop, (int(shard_idx)+1) * n_vocab)):
            logging.info(f"Generating {i}th sample for {shard_idx}/{num_shards} shard")
            input_ids = torch.tensor([[i]]).to(device)
            outputs1 = model.generate(input_ids, do_sample=False, max_length=j)
            outputs = model.generate(outputs1, do_sample=True, max_length=1024)
            gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            text_dict = {"text" : gen_text[0]}
            with open(f"{save_dir}/gen.chunk.{str(shard_idx).zfill(2)}-{str(num_shards).zfill(2)}.jsonl", "a") as f:
                f.write(json.dumps(text_dict))
                f.write('\n')

if __name__ == "__main__":
    fire.Fire(main)