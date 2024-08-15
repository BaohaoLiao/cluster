import numpy as np
import time
import faiss


def train_kmeans(x, k, ngpu):
    d = x.shape[1]
    niter = 5
    verbose = True
    kmeans = faiss.Kmeans(d, k, niter=niter, verbose=verbose, gpu=ngpu)
    kmeans.train(x)


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
