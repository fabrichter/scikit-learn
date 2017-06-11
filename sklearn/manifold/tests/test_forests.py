from sklearn.manifold.forests import ManifoldForest, _Tree
from sklearn.ensemble.forest import RandomForestClustering
import numpy as np
import matplotlib.pyplot as plt

def test_tree():

    data = np.array([
        [1,1], [1,-1], [-1, 1], [-1, -1]
    ], dtype=float)
    data = np.tile(data, (20,1))
    data += 0.3*np.random.rand(*data.shape, )

    if False:
        plt.figure()
        plt.scatter(*zip(*data))
        plt.show()

    # tree = _Tree(2, 2, 5) #using all features and full depth
    # tree.fit(data,)
    # clusters = tree.predict(data)

    forest = RandomForestClustering(n_estimators=5, max_depth=2)
    clusters = forest.fit_transform(data)

    print(clusters)

    if True:
        plt.figure()
        plt.scatter(*zip(*data), c=clusters.flatten())
        plt.show()


test_tree()

