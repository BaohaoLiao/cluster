import fire
import copy
import logging
from tqdm import tqdm
from collections import OrderedDict
import faiss
import torch
import transformers
import numpy as np

logger = logging.getLogger(__name__)

def split_array(array, n):
    # Calculate the length of each chunk
    k, m = divmod(len(array), n)
    return [array[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def run_faiss_gpu(x, num_clusters, niter=20, verbose=True, nredo=1, ngpu=1, use_fp16=False):
    vector_dim = x.shape[1]

    # Perform the clustering
    kmeans = faiss.Clustering(vector_dim, num_clusters)
    kmeans.niter = niter  # Number of iterations
    kmeans.verbose = verbose  # Print progress
    kmeans.nredo = nredo
    kmeans.max_points_per_centroid = 10000000 # otherwise the kmeans implementation sub-samples the training set
    
    res = [faiss.StandardGpuResources() for i in range(ngpu)]
    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = use_fp16
        cfg.device = i
        flat_config.append(cfg)
    if ngpu == 1:
        index = faiss.GpuIndexFlatL2(res[0], vector_dim, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], vector_dim, flat_config[i]) for i in range(ngpu)]
        index = faiss.IndexProxy()
        for sub_index in indexes:
            index.addIndex(sub_index)

    # Define the index and GPU resources for the clustering object
    kmeans.train(x, index)
    # Get the resulting centroids
    centroids = faiss.vector_to_array(kmeans.centroids).reshape(num_clusters, vector_dim)
    # Get the cluster assignment for each vector
    _, assignments = index.search(x, 1)
    assignments = assignments.flatten()
    rec_x = centroids[assignments]
    return rec_x


def main(model_name_or_path: str, save_dir: str, ngpu: int, bit: float, size: int=16):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='cpu',
    )

    if bit == 4.14: # size = 4
        layers = {
            'q_proj.weight': [1, 38000],
            'k_proj.weight': [1, 38000],
            'v_proj.weight': [1, 38000],
            'o_proj.weight': [1, 38000],
            'gate_proj.weight': [2, 51000],
            'up_proj.weight': [2, 51000],
            'down_proj.weight': [2, 51000],
        }
    elif bit == 3: # size = 6
        layers = {
            'q_proj.weight': [1, 57000],
            'k_proj.weight': [1, 57000],
            'v_proj.weight': [1, 57000],
            'o_proj.weight': [1, 57000],
            'gate_proj.weight': [3, 51000],
            'up_proj.weight': [3, 51000],
            'down_proj.weight': [3, 51000],
        }
    elif bit == 2.28: # size = 8
        layers = {
            'q_proj.weight': [1, 38000],
            'k_proj.weight': [1, 38000],
            'v_proj.weight': [1, 38000],
            'o_proj.weight': [1, 38000],
            'gate_proj.weight': [2, 50000],
            'up_proj.weight': [2, 50000],
            'down_proj.weight': [2, 50000],
        }
    elif bit == 2.14: # size = 8
        layers = {
            'q_proj.weight': [1, 19000],
            'k_proj.weight': [1, 19000],
            'v_proj.weight': [1, 19000],
            'o_proj.weight': [1, 19000],
            'gate_proj.weight': [1, 50000],
            'up_proj.weight': [1, 50000],
            'down_proj.weight': [1, 50000],
        }
    elif bit == 2: # size = 9
        layers = {
            'q_proj.weight': [1, 27000],
            'k_proj.weight': [1, 27000],
            'v_proj.weight': [1, 27000],
            'o_proj.weight': [1, 27000],
            'gate_proj.weight': [2, 35000],
            'up_proj.weight': [2, 35000],
            'down_proj.weight': [2, 35000],
        }
    
    
    ws = OrderedDict()
    for k, v in model.state_dict().items():
        if ".".join(k.split(".")[-2:]) in layers.keys():
            ws[k] = copy.deepcopy(v.transpose(1, 0).contiguous())

    rec_ws = OrderedDict()
    for k, w in tqdm(ws.items()):
        logger.info(f"Reconstruct {k} ...\n")
        ori_shape = w.shape
        new_k = ".".join(k.split(".")[-2:])

        deficiency = ori_shape[1] % size
        if deficiency > 0:
            deficiency = size - deficiency
            pad_zeros = torch.zeros((ori_shape[0], deficiency), dtype=w.dtype)
            w = torch.cat((w, pad_zeros), dim=1)

        reshaped_w = w.view(-1, size).numpy()
        num_split = layers[new_k][0]
        nclusters = layers[new_k][1]
        if num_split > 1:
            reshaped_ws = split_array(reshaped_w, num_split)
            rec_w_tmps = []
            for i in range(num_split):
                rec_w_tmps.append(
                    run_faiss_gpu(
                        reshaped_ws[i], nclusters, niter=20, verbose=True, nredo=1, ngpu=ngpu
                    )
                )
            rec_w = np.vstack(rec_w_tmps)
        else:
            rec_w = run_faiss_gpu(reshaped_w, nclusters, niter=20, verbose=True, nredo=1, ngpu=ngpu)

        if deficiency > 0:
            rec_ws[k] = torch.from_numpy(rec_w).view(ori_shape[0], -1)[:, :-deficiency]
        else:
            rec_ws[k] = torch.from_numpy(rec_w).view(ori_shape[0], ori_shape[1])
        logger.info("\n")

    for k in rec_ws.keys():
        logger.info(f"Reconstruction error of {k}: {torch.norm(rec_ws[k] - ws[k])}")

    for k, w in rec_ws.items():
        rec_ws[k] = w.transpose(1, 0).to(torch.bfloat16)

    logger.info("Saving model ...")
    torch.save(rec_ws, f'{save_dir}/cluster_model.pth')


if __name__=="__main__":
    fire.Fire(main)