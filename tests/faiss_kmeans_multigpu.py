import numpy as np
import time
import faiss


def train_kmeans(x, k, ngpu):
    d = x.shape[1]
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.niter = 5

    # otherwise the kmeans implementation sub-samples the training set
    clus.max_points_per_centroid = 10000000

    res = [faiss.StandardGpuResources() for i in range(ngpu)]

    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = True
        cfg.device = i
        flat_config.append(cfg)

    if ngpu == 1:
        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i]) for i in range(ngpu)]
        index = faiss.IndexReplicas()
        for sub_index in indexes:
            index.addIndex(sub_index)

    # perform the training
    clus.train(x, index)
    centroids = faiss.vector_float_to_array(clus.centroids)

    obj = faiss.vector_float_to_array(clus.obj)
    print("final objective: %.4g" % obj[-1])

    return centroids.reshape(k, d)


def main():
    x = np.random.rand(10000, 3)
    k = x.shape[0] // 40

    print("run")
    t0 = time.time()
    train_kmeans(x, k, 1)
    t1 = time.time()

    print("total runtime: %.3f s" % (t1 - t0))


if __name__ == "__main__":
    main()
