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
ct = [0.45]
# ct = [0.075]
ct2 = 0.42


# HELPER FUNCTIONS ---------------------------------------------------------------------------------------------------

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


def tdplot(data, raw=False):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    # colors = np.array(['#ff7f00', '#377eb8'])
    colors = np.array(['#377eb8', '#ff7f00'])
    if raw:
        colors = np.array(['#ff7f00', '#377eb8'])
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
    # plt.savefig('../output/smtp/rsf_smtp_anomalies_tb.png')
    plt.show()


# DATASETS ------------------------------------------------------------------------------------------------------------

diabetes_data = pd.read_csv('../data/diabetes.csv')
# diabetes_data_no_class = diabetes_data.iloc[:, :-1]
diabetes_answers = diabetes_data.iloc[:, -1].tolist()
diabetes_data_normalized = (diabetes_data - diabetes_data.min()) / (diabetes_data.max() - diabetes_data.min())
diabetes_data_no_class = diabetes_data_normalized.iloc[:, :-1]
for inde, a in enumerate(diabetes_answers):
    if a == 1:
        diabetes_answers[inde] = -1
    else:
        diabetes_answers[inde] = 1
diabetes_data_list = diabetes_data_normalized.values.tolist()
print(diabetes_data_normalized)


# END OF DATASETS -----------------------------------------------------------------------------------------------------


def main(plot_it=False):
    for j in ct:
        st = time.time()
        results_if = IsolationForest(contamination=j, n_estimators=100).fit_predict(diabetes_data_no_class)
        # plot_res = []
        # for g, val in enumerate(smtp_data):
            # val.append(results_if[g])
            # plot_res.append(val)
        ctr = 0
        ctr2 = 0
        ctr3 = 0
        for ind, res in enumerate(results_if):
            if diabetes_answers[ind] == -1 and res == -1:
                ctr += 1
            if res == -1:
                ctr2 += 1
        print("Contamination: ", j)
        print("True Positives: ", ctr, "percentage correct: ", ctr / int(j * len(diabetes_answers)))
        print("Total Positives: ", ctr2)
        print("Anomalies in data: ", ctr3)
        print("ROC-AUC Score: ", roc_auc_score(diabetes_answers, results_if))
        # print("results if: ", results_if)
        print("Running time: ", time.time() - st)
        # tdplot(plot_res, raw=True)
        # raise SystemExit(0)

    start_time = time.time()
    clf = RSForest(contamination=ct2, k=100, points=diabetes_data_list, granularity=10, sample_size=256)
    # print(smtp_data_answers)
    result, answers_rsf = clf.fit_predict()
    # print(answers_rsf)
    answers_rs = [-1 if item in [1.0] else 1 for item in answers_rsf]
    results = [p.is_outlier for p in result]
    filtered_results = [p for p in result if p.is_outlier == -1]
    filtered_results = sorted(filtered_results[:int(len(result) * ct2)], key=lambda x: x.anomaly_score)
    table = PrettyTable((['Coordinates', 'Anomaly Score']))
    for res in filtered_results:
        table.add_row([res.coordinates[-1], res.anomaly_score])
    print(table)
    c = 0
    for it in filtered_results:
        if it.coordinates[-1] not in [1]:
            c += 1
    print(answers_rs)
    print(results)
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


cProfile.run('main(plot_it=False)')
