import fire
import copy
import logging
from tqdm import tqdm
from collections import OrderedDict
import faiss
import torch
import transformers

logger = logging.getLogger(__name__)


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
    return centroids, assignments


def main(model_name_or_path: str, save_dir: str, ngpu: int, size: int=16, nclusters: int=64000):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='cpu',
    )

    layers = ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight', 'gate_proj.weight', 'up_proj.weight', 'down_proj.weight']
    
    ws = OrderedDict()
    for k, v in model.state_dict().items():
        if ".".join(k.split(".")[-2:]) in layers:
            ws[k] = copy.deepcopy(v.transpose(1, 0).contiguous())

    rec_ws = OrderedDict()
    for k, w in tqdm(ws.items()):
        logger.info(f"Reconstruct {k} ...\n")
        ori_shape = w.shape
        reshaped_w = w.view(-1, size)
        rec_w = run_faiss_gpu(reshaped_w.numpy(), nclusters, niter=20, verbose=True, nredo=1, ngpu=ngpu)
        rec_ws[k] = torch.from_numpy(rec_w).view(ori_shape[0], ori_shape[1])
        logger.info("\n")

    for k in rec_ws.keys():
        logger.info(f"Reconstruction error of {k}: {torch.norm(rec_ws[k] - ws[k])}")

    for k, w in rec_ws.items():
        rec_ws[k] = w.transpose(1, 0)

    logger.info("Saving model ...")
    torch.save(rec_ws, f'{save_dir}/cluster_model.pth')


if __name__=="__main__":
    fire.Fire(main)