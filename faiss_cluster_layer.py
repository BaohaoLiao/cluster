import fire
import copy
import logging
from tqdm import tqdm
from collections import OrderedDict
import faiss
import torch
import transformers
from safetensors.torch import save_file

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

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


def main(model_name_or_path: str, save_dir: str, ngpu: int, size: int=4, nclusters: int=65500):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='cpu',
    )

    layers = ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight', 
              'gate_proj.weight', 'up_proj.weight', 'down_proj.weight']
    
    ws = OrderedDict()
    for k, v in model.state_dict().items():
        if ".".join(k.split(".")[-2:]) in layers:
            ws[k] = copy.deepcopy(v.transpose(1, 0).contiguous()) # transpose has better performance

    logging.info(f"{'-'*20} Clustering {'-'*20}")
    cluster_model = OrderedDict()
    for k, w in tqdm(ws.items()):
        logging.info("\n")
        logging.info(f"cluster {k} ...\n")
        ori_shape = w.shape

        deficiency = ori_shape[1] % size
        if deficiency > 0:
            deficiency = size - deficiency
            pad_zeros = torch.zeros((ori_shape[0], deficiency), dtype=w.dtype)
            w = torch.cat((w, pad_zeros), dim=1)

        reshaped_w = w.view(-1, size)
        centroids, indices = run_faiss_gpu(reshaped_w.numpy(), nclusters, niter=20, verbose=True, nredo=1, ngpu=ngpu)

        new_k = ".".join(k.split(".")[:-1])
        cluster_model[new_k + ".cluster"] = torch.from_numpy(centroids).to(torch.bfloat16)
        cluster_model[new_k + ".index"] = torch.from_numpy(indices).to(torch.int32) # save_file doesn't support saving uint16

    # Add the missing layers
    logging.info(f"{'-'*20} Adding missing layers {'-'*20}")
    for k, v in model.state_dict().items():
        if ".".join(k.split(".")[-2:]) not in layers:
            logging.info(f"Add {k}")
            cluster_model[k] = v

    # Save for easy loading
    logging.info(f"{'-'*20} Saving model {'-'*20}")
    save_file(cluster_model, f'{save_dir}/model.safetensors', metadata={"format": "pt"})
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.save_pretrained(save_dir)
    config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    config.num_clusters = nclusters
    config.cluster_dim = size
    config.save_pretrained(save_dir)

if __name__=="__main__":
    fire.Fire(main)