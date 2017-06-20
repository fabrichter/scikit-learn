import numpy as np
import sklearn.manifold.forestsNaive as forestNaive
from sklearn.manifold.forests import ManifoldForest as forestProper
from sklearn.manifold import SpectralEmbedding
from time import time
from sklearn import datasets
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def evaluateClustering(tree, nclusters, nelements, plotTrees=3):
    np.random.seed(134495995)
    plt.ion()
    center = np.random.rand(nclusters, 2) * 10

    clusterIDS = np.random.choice(nclusters, size=(nelements,))

    noise = np.random.normal(0, 0.33, size=(nelements, 2))

    result = center[clusterIDS] + noise

    fig = plt.figure(figsize=(8, 8))
    plt.suptitle("clustering with tree {}".format(tree))
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("ground truth")
    ax.scatter(result[:, 0], result[:, 1], c=clusterIDS)

    tree.fit_transform(result)

    treeClustering = tree.apply(result)

    #     print(treeClustering)

    for i in range(plotTrees):
        ax = fig.add_subplot(2, 2, 2 + i)
        ax.set_title("clustering of tree {}".format(i))
        ax.scatter(result[:, 0], result[:, 1], c=treeClustering[:, i], cmap=plt.cm.Paired)

    # plt.show()
    print()

evaluateClustering(forestNaive.AffinityForestNaiveRecursive(10, 2, num_options=23, num_features=2), 4, 100)