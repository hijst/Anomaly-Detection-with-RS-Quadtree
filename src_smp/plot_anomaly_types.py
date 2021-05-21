import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

raw_input_data = make_blobs(n_samples=380, n_features=2, centers=[[2, 2], [4, 2], [7, 3], [9, 3.5], [11, 4], [15, 5], [17, 5.5],
                                                              [20, 6], [18, 5], [10, 4], [7, 3], [5, 2.5]], cluster_std=4)[0].tolist()
collective_anomaly = [[35, 10], [34, 9], [33, 9.5], [32, 10], [33, 10.5], [32, 9], [31, 9]]

input_data = [i for i in raw_input_data if i[1] > 0]

for item in collective_anomaly:
    input_data.append(item)


def plot_anomaly_types(data):
    plt.figure()
    ax = plt.axes()
    xs = [x[0] for x in data]
    ys = [y[1] for y in data]
    ax.scatter(xs, ys, s=16)
    # plt.savefig('../output/anomaly_types.pdf')
    plt.show()


plot_anomaly_types(input_data)
