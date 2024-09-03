import os
import sys
import argparse
import logging
import random
import time
import numpy as np
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from calibrate.cal import cal
from calibrate.data_utils import get_loaders
from calibrate.evaluate import evaluate
import models


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


MODEL_FAMILY = [
    "llama",
    "mistral",
    "meta", #for llama3
]

"""
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Initialization
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ori_config = AutoConfig.from_pretrained(args.ori_model_name_or_path, attn_implementation=args.attn_implementation)
    tokenizer = AutoTokenizer.from_pretrained(args.ori_model_name_or_path, use_fast=False, legacy=False)
    ori_model = AutoModelForCausalLM.from_pretrained(args.ori_model_name_or_path, config=ori_config, device_map='cpu', torch_dtype=torch.bfloat16)
    
    clus_config = AutoConfig.from_pretrained(args.clus_model_name_or_path)
    clus_model = models.CustomLlamaForCausalLM(clus_config).to(torch.bfloat16)
    logging.info(f"Load state dicts from {args.state_dict_path}")
    state_dicts = torch.load(args.state_dict_path, map_location="cpu")
    clus_model.load_state_dict(state_dicts, strict=True)

    assert args.seqlen <= ori_config.max_position_embeddings, "The sequence length of calibration samples exceed the model's"

    ori_model.eval()
    clus_model.eval()
    logging.info(ori_model)
    logging.info(clus_model)

    # Quantization 
    logging.info("=== start calibration ===")
    tick = time.time()
    dataloader, _ = get_loaders(
        args.calib_dataset,
        tokenizer,
        args.cache_dir,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
    )

    # Only tune vector bank
    for n, p in clus_model.named_parameters():
        if "vector_bank" in n:
            logging.info(f"{n} is trainable")
            p.requires_grad = True
        else:
            p.requires_grad = False

    for n, p in ori_model.named_parameters():
        p.requires_grad = False


    cal(ori_model, clus_model, args, dataloader, logging=logging)
    logging.info(f"Time for calibration: {time.time() - tick} s")
    evaluate(clus_model, tokenizer, args, logging)

    if not args.resume:
        logging.info(f"Save clus model.")
        clus_model.save_pretrained(args.save_dir, safe_serialization=False)
        tokenizer.save_pretrained(args.save_dir)
        clus_config.save_pretrained(args.save_dir)
"""
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Initialization
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ori_config = AutoConfig.from_pretrained(args.ori_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.ori_model_name_or_path) #, use_fast=False, legacy=False)
    ori_model = AutoModelForCausalLM.from_pretrained(args.ori_model_name_or_path, config=ori_config, device_map='cpu', torch_dtype=torch.bfloat16)
    
    clus_config = AutoConfig.from_pretrained(args.clus_model_name_or_path)
    clus_model = models.ClusterLayerLlamaForCausalLM.from_pretrained(
        args.clus_model_name_or_path,
        config=clus_config,
        device_map='cpu', 
        torch_dtype=torch.bfloat16
    )
    """
    clus_model = models.CustomLlamaForCausalLM(clus_config).to(torch.bfloat16)
    logging.info(f"Load state dicts from {args.state_dict_path}")
    state_dicts = torch.load(args.state_dict_path, map_location="cpu")
    clus_model.load_state_dict(state_dicts, strict=True)
    """
    assert args.seqlen <= ori_config.max_position_embeddings, "The sequence length of calibration samples exceed the model's"

    ori_model.eval()
    clus_model.eval()
    #logging.info(ori_model)
    logging.info(clus_model)

    # Quantization 
    logging.info("=== start calibration ===")
    tick = time.time()
    dataloader, _ = get_loaders(
        args.calib_dataset,
        tokenizer,
        args.cache_dir,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
    )

    # Only tune cluster
    for n, p in clus_model.named_parameters():
        if "cluster" in n:
            logging.info(f"{n} is trainable")
            p.requires_grad = True
        else:
            p.requires_grad = False

    for n, p in ori_model.named_parameters():
        p.requires_grad = False


    cal(ori_model, clus_model, args, dataloader, logging=logging)
    logging.info(f"Time for calibration: {time.time() - tick} s")
    evaluate(clus_model, tokenizer, args, logging)

    if not args.resume:
        logging.info(f"Save cluster model.")
        clus_model.save_pretrained(args.save_dir) #, safe_serialization=False)
        tokenizer.save_pretrained(args.save_dir)
        clus_config.save_pretrained(args.save_dir)


def arg_parse():
    parser = argparse.ArgumentParser(description="Quantize a model")
    parser.add_argument("--seed", type=int, default=42)
    # Model
    parser.add_argument("--ori_model_name_or_path", type=str, required=True)
    parser.add_argument("--clus_model_name_or_path", type=str, required=True)
    parser.add_argument("--state_dict_path", type=str, default=None)
    parser.add_argument(
        "--attn_implementation", type=str, required=False, default="eager", choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation that the model works with",
    )
    # Calibration data
    parser.add_argument("--calib_dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb", "c4", "mix", "pile"])
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length of calibration sample")
    # Training
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--aug_loss", default=False, action="store_true", help="calculate additional loss with same quant input")
    # Output
    parser.add_argument("--cache_dir", default="./cache", type=str, help="Cache dir of dataset, leading to faster debug")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_dir", default="./models/", type=str, help="Direction for saving model")
    # Other
    parser.add_argument("--eval_ppl", default=False, action="store_true")
    parser.add_argument("--limit", type=int, default=-1, help="Number of samples in evaluation for debug purpose.")

    args = parser.parse_args()
    return args


def cli_main():
    args = arg_parse()
    logging.info(sys.argv)
    logging.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()