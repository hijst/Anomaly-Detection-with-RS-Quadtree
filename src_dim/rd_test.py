import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.datasets import make_blobs, make_swiss_roll
from src_dim.rd_forest import RDForest
from sklearn import datasets
import time

ct = 0.06


def plot(data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    colors = np.array(['#ff7f00', '#377eb8'])
    xs = [x.coordinates[0] for x in data]
    ys = [y.coordinates[1] for y in data]
    zs = [z.coordinates[2] for z in data]
    ax.scatter3D(xs, ys, zs, c=colors[[(p.is_outlier + 1) // 2 for p in data]])
    # plt.savefig('../output/3dplot.pdf')
    plt.show()


# DATASETS ------------------------------------------------------------------------------------------------------------
big_small_blob = make_blobs(centers=[[0, 0, 0], [4, 4, 4]], n_samples=[950, 50],
                            n_features=2, cluster_std=[1.0, 2.0])[0]

# swiss_roll = make_swiss_roll(n_samples=500, noise=0.1, random_state=None)[0]

# iris = datasets.load_iris()
# iris_data = iris.data[:, :3]

# END OF DATASETS -----------------------------------------------------------------------------------------------------


data = big_small_blob

start_time = time.time()
clf = RDForest(contamination=ct, k=10, points=data)
result = clf.fit_predict()
filtered_results = [p for p in result if p.is_outlier == 0]
filtered_results = sorted(filtered_results[:int(len(result) * ct)], key=lambda x: x.anomaly_score)
table = PrettyTable((['Coordinates', 'Anomaly Score']))
for res in filtered_results:
    table.add_row([res.coordinates, res.anomaly_score])
print(table)
print(data[:50])
print("--- running time: {:.2f} seconds ---".format(time.time() - start_time))
plot(result)
