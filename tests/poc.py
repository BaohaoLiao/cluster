import copy
from collections import OrderedDict
import fire
import faiss
import torch
import transformers


class args:
    model_name = "/mnt/nushare2/data/baliao/PLLMs/meta-llama/Llama-2-7b-hf"
    device = "cpu"

def run_faiss_gpu(x, num_clusters, niter=20, verbose=True, nredo=1, ngpu=1, use_fp16=False):
    vector_dim = x.shape[1]

    # Perform the clustering
    kmeans = faiss.Clustering(vector_dim, num_clusters)
    kmeans.niter = niter  # Number of iterations
    kmeans.verbose = verbose  # Print progress
    kmeans.nredo = nredo
    
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

def combine(xs, size=16):
    outs = []
    for k, v in xs.items():
        outs.append(v.view(-1, size))
    return torch.cat(outs, dim=0)

def split_combine(weights, labels, size=16):
    rec_labels = OrderedDict()
    start = 0
    for k, v in weights.items():
        end = v.shape[0] * v.shape[1] // size
        end = start + end
        rec_labels[k] = labels[start:end]
        start = end
    return rec_labels

def main(save_dir: str="/data/chatgpt/data/baliao/cluster/00_prepare/cluster/tests", ngpu: int=2):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device,
    )

    layers = ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight', 'gate_proj.weight', 'up_proj.weight', 'down_proj.weight']
    cluster_model = OrderedDict()
    cluster_labels = OrderedDict()

    for i in range(1): # model.config.num_hidden_layers):
        blocks = OrderedDict()
        for k, v in model.state_dict().items():
            if f"model.layers.{i}." in k:
                if ".".join(k.split(".")[-2:]) in layers:
                    blocks[k] = copy.deepcopy(v).transpose(1, 0).contiguous()
                    
        combined_blocks = combine(blocks, size=16)
        centroids, labels = run_faiss_gpu(combined_blocks, combined_blocks.shape[0]//8, niter=1, verbose=True, nredo=1, ngpu=ngpu)
        cluster_model[f"layers.{i}"] = centroids
        cluster_labels.update(split_combine(blocks, labels, size=16))

    torch.save(cluster_model, f'{save_dir}/cluster_model.pth')
    torch.save(cluster_labels, f'{save_dir}/cluster_label.pth')


if __name__=="__main__":
    fire.Fire(main)