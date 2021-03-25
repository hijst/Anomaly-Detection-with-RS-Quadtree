import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.datasets import make_blobs
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
ct = [0.13]
# ct = [0.075]
ct2 = 0.125


# HELPER FUNCTIONS ---------------------------------------------------------------------------------------------------

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


def tdplot(data):
    plt.figure()
    ax = plt.axes(projection='3d')
    colors = np.array(['#ff7f00', '#377eb8'])
    xs = [x.coordinates[0] for x in data]
    ys = [y.coordinates[1] for y in data]
    zs = [z.coordinates[2] for z in data]
    ax.scatter3D(xs, ys, zs, c=colors[[(p.is_outlier + 1) // 2 for p in data]])
    # plt.savefig('../output/massive_data_no_noise.pdf')
    plt.show()


# DATASETS ------------------------------------------------------------------------------------------------------------
big_small_blob = make_blobs(centers=[[0, 0, 0], [12, 12, 12]], n_samples=[8000, 1000],
                            n_features=3, cluster_std=[4.0, 2.0])[0]

shuttle_data = pd.read_csv('../data/shuttle.csv', delim_whitespace=True, header=None)
shuttle_data_no_class = shuttle_data.iloc[:, :-1]
normalized_shuttle = (shuttle_data - shuttle_data.min()) / (shuttle_data.max() - shuttle_data.min())
answers_shuttle_raw = shuttle_data.iloc[:, -1]
answers_shuttle = [1 if item in [1, 4] else -1 for item in answers_shuttle_raw]
# print(answers_shuttle[:100])
# print("points for RSQT: ", normalized_shuttle.values.tolist()[:100])
real_shuttle_data = normalized_shuttle.values.tolist()

satellite_data_answers_raw = pd.read_pickle('../data/sat_full.pkl')
satellite_data_answers = satellite_data_answers_raw.values.tolist()
satellite_data = satellite_data_answers_raw.iloc[:, :-1]
satellite_answers_raw = satellite_data_answers_raw.iloc[:, -1:]
satellite_answers = []
for sa in satellite_answers_raw[36]:
    if sa in [1, 3, 4, 5, 7]:
        satellite_answers.append(1)
    else:
        satellite_answers.append(-1)
print(satellite_answers)

# END OF DATASETS -----------------------------------------------------------------------------------------------------

raw_data = big_small_blob
dims = len(raw_data[0])
random_data = (np.random.rand(1000, dims) - 0.5) * 40
data = np.concatenate((big_small_blob, random_data))


# print("part of anomalies in dataset: ", count_anomaly / len(real_data_small), "number of anomalies: ", count_anomaly)
print("anomalies in dataset: ", answers_shuttle.count(-1), "percentage: ",
      answers_shuttle.count(-1) / len(answers_shuttle))


def main(plot_it=False):
    for j in ct:
        st = time.time()
        results_if = IsolationForest(contamination=j, n_estimators=100).fit_predict(satellite_data)
        ctr = 0
        ctr2 = 0
        ctr3 = 0
        for ind, res in enumerate(results_if):
            if satellite_answers[ind] == -1 and res == -1:
                ctr += 1
            if res == -1:
                ctr2 += 1
        print("Contamination: ", j)
        print("True Positives: ", ctr, "percentage correct: ", ctr / int(j * len(satellite_answers)))
        print("Total Positives: ", ctr2)
        print("ROC-AUC Score: ", roc_auc_score(satellite_answers, results_if))
        print("results if: ", results_if)
        print("Running time: ", time.time() - st)
        # raise SystemExit(0)

    start_time = time.time()
    clf = RSForest(contamination=ct2, k=25, points=satellite_data_answers, granularity=5, sample_size=64)
    result, answers_rsf = clf.fit_predict()
    # print(answers_rsf)
    answers_rs = [1 if item in [1, 3, 4, 5, 7] else -1 for item in answers_rsf]
    results = [p.is_outlier for p in result]
    filtered_results = [p for p in result if p.is_outlier == -1]
    filtered_results = sorted(filtered_results[:int(len(result) * ct2)], key=lambda x: x.anomaly_score)
    table = PrettyTable((['Coordinates', 'Anomaly Score']))
    for res in filtered_results:
        table.add_row([res.coordinates[-1], res.anomaly_score])
    print(table)
    c = 0
    for it in filtered_results:
        if it.coordinates[-1] not in [1, 3, 4, 5, 7]:
            c += 1
    print("percentage of anomalies in final results: ", c / len(filtered_results), "anomalies found: ",
          c, "normal points found: ", len(filtered_results) - c)
    print("ROC-AUC score: ", roc_auc_score(answers_rs, results))
    print("--- running time: {:.2f} seconds ---".format(time.time() - start_time))

    if plot_it:
        tdplot(result)

    # base = clf.trees[0].root
    # dt = build_tree(base)
    # dt.render()
    # print("rendered")


main(plot_it=False)
