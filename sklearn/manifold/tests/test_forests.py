from sklearn.manifold.forests import ManifoldForest, _Tree
import numpy as np
import matplotlib.pyplot as plt


def test_maniflod():
    data = np.random.rand(10,5)

    mf = ManifoldForest(10, 2, 9, 5)

    embeding = mf.fit_transform(data, 2)

    print(embeding.shape)

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

    tree = _Tree(4, 2, 10) #using all features and full depth
    tree.fit(data,)
    clusters = tree.predict(data)

    print(clusters)

    if True:
        plt.figure()
        plt.scatter(*zip(*data), c=clusters)
        plt.show()


test_maniflod()

