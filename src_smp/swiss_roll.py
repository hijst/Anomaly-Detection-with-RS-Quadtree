import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from prettytable import PrettyTable
from sklearn.datasets import make_blobs, make_swiss_roll, make_s_curve
from src_smp.rs_forest import RSForest
import hdf5storage
import pandas as pd
from sklearn import datasets
import time
import cProfile
from graphviz import Digraph
import os
import csv
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from scipy.io import loadmat

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

# contamination in dataset
ct = [0.02]
# ct = [0.075]
ct2 = 0.02


# HELPER FUNCTIONS ---------------------------------------------------------------------------------------------------


def rotate(angle):
    ax.view_init(elev=0, azim=angle)

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def build_tree(tree, dot=None):
    if dot is None:
        dot = Digraph(directory='../graph_output')
        dot.node(name=str([round(c, 2) for c in tree.hc.center]))
    if len(tree.children) > 0:
        dot.node(name=str([round(d, 2) for d in tree.children[0].hc.center]))
        dot.edge(str([round(e, 2) for e in tree.hc.center]), str([round(f, 2) for f in tree.children[0].hc.center]))
        dot = build_tree(tree.children[0], dot=dot)
        dot.node(name=str([round(g, 2) for g in tree.children[1].hc.center]))
        dot.edge(str([round(h, 2) for h in tree.hc.center]), str([round(i, 2) for i in tree.children[1].hc.center]))
        dot = build_tree(tree.children[1], dot=dot)
    return dot


def plot(ddata):
    plt.figure()
    ax = plt.axes()
    colors = np.array(['#377eb8', '#ff7f00'])
    xs = [x.coordinates[0] for x in ddata]
    ys = [y.coordinates[1] for y in ddata]
    ax.scatter(xs, ys, s=16, c=colors[[(p.is_outlier + 1) // 2 for p in ddata]])
    # plt.savefig('../output/dataset_iforest_example.pdf')
    plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

def tdplot(data, raw=False):
    colors = np.array(['#377eb8', '#ff7f00'])
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    if raw:
        xs = [x[0] for x in data]
        ys = [y[1] for y in data]
        zs = [z[2] for z in data]
        ax.scatter3D(xs, ys, zs, c=colors[[(int(p[-1]) - 1) // 2 for p in data]])
    else:
        colors = np.array(['#ff7f00', '#377eb8'])
        xs = [x.coordinates[0] for x in data]
        ys = [y.coordinates[1] for y in data]
        zs = [z.coordinates[2] for z in data]
        ax.scatter3D(xs, ys, zs, c=colors[[(p.is_outlier + 1) // 2 for p in data]])
    # plt.savefig('../output/swiss roll/swiss_roll_rsf_wbg.png')
    ax.set_axis_off()
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=1)
    rot_animation.save('../output/rotation_def.gif', dpi=180, writer='imagemagick')
    plt.show()


# DATASETS ------------------------------------------------------------------------------------------------------------
big_small_blob = make_blobs(centers=[[0, 0, 0], [12, 12, 12]], n_samples=[8000, 1000],
                            n_features=3, cluster_std=[4.0, 2.0])[0]

# swiss_roll = make_swiss_roll(n_samples=4950)[0].tolist()
swiss_roll = datasets.make_s_curve(4900, random_state=0)[0].tolist()
iris = datasets.load_iris()
iris_data = iris.data[:, :3].tolist()

# END OF DATASETS -----------------------------------------------------------------------------------------------------

raw_data = swiss_roll
dims = len(raw_data[0])
random_data_x = (np.random.rand(100) - 0.5) * 3
random_data_y = (np.random.rand(100)) * 2
random_data_z = (np.random.rand(100) - 0.5) * 3
random_data = np.column_stack((random_data_x, random_data_y, random_data_z))
random_data_list = random_data.tolist()
data = np.concatenate((swiss_roll, random_data))

for item in random_data_list:
    swiss_roll.append(item)


def main(plot_it=False):
    for j in ct:
        results_if = IsolationForest(contamination=j, n_estimators=100, max_samples=512).fit_predict(swiss_roll)
        plot_res = []
        for g, val in enumerate(swiss_roll):
            val.append(results_if[g])
            plot_res.append(val)
        #tdplot(plot_res, raw=True)

    clf = RSForest(contamination=ct2, k=100, points=swiss_roll, granularity=100, sample_size=1024)
    result, answers_rsf = clf.fit_predict()
    filtered_results = [p for p in result if p.is_outlier == -1]
    filtered_results = sorted(filtered_results[:int(len(result) * ct2)], key=lambda x: x.anomaly_score)
    table = PrettyTable((['Coordinates', 'Anomaly Score']))
    for res in filtered_results:
        table.add_row([res.coordinates[-1], res.anomaly_score])
    print(table)

    if plot_it:
        tdplot(result)

    # base = clf.trees[0].root
    # dt = build_tree(base)
    # dt.render()
    # print("rendered")


main(plot_it=True)
