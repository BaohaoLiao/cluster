import random
import torch
import time
import torch.nn as nn
import transformers
from datasets import load_dataset
from tqdm import tqdm
from collections import OrderedDict
import functools
import peft
from sklearn.cluster import KMeans
import faiss

class args:
    model_name = "/mnt/nushare2/data/baliao/PLLMs/meta-llama/Llama-2-7b-hf"
    device = "cpu"
    nsamples = 128
    seqlen = 2048

def main():
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device,
    )

    weight = model.state_dict()['model.layers.31.self_attn.q_proj.weight'].cpu()
    x = weight.view(-1, 2)

    verbose = True
    d = x.shape[1]
    niter = 20

    start = time.time()
    kmeans = faiss.Kmeans(d, x.size(0)//64, niter=niter, verbose=verbose, gpu=True)
    kmeans.train(x)
    end = time.time()
    print(f"Using time: {end - start}s")


if __name__ == "__main__":
    main()