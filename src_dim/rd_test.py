import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.datasets import make_blobs, make_swiss_roll
from src_dim.rd_forest import RDForest
import pandas as pd
from sklearn import datasets
import time
import cProfile
from graphviz import Digraph
import os
import csv
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

# contamination in dataset
ct = [0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.1]
#ct = [0.075]
ct2 = 0.1


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
    colors = np.array(['#377eb8', '#ff7f00'])
    xs = [x.coordinates[0] for x in data]
    ys = [y.coordinates[1] for y in data]
    zs = [z.coordinates[2] for z in data]
    ax.scatter3D(xs, ys, zs, c=colors[[(p.is_outlier + 1) // 2 for p in data]])
    # plt.savefig('../output/massive_data_no_noise.pdf')
    plt.show()


# DATASETS ------------------------------------------------------------------------------------------------------------
big_small_blob = make_blobs(centers=[[0, 0, 0], [12, 12, 12]], n_samples=[800, 100],
                            n_features=3, cluster_std=[4.0, 2.0])[0]

with open('../data/mammography.csv', newline='') as f:
    reader = csv.reader(f)
    real_data_full = list(reader)

shuttle_data = pd.read_csv('../data/shuttle.csv', delim_whitespace=True, header=None)
shuttle_data_no_class = shuttle_data.iloc[:, :-1]
answers_shuttle_raw = shuttle_data.iloc[:, -1]
answers_shuttle = [1 if item in [1, 4] else -1 for item in answers_shuttle_raw]
print(answers_shuttle[:100])
print("points for RSQT: ", shuttle_data.values.tolist()[:100])

real_data = np.core.defchararray.replace(np.array(real_data_full[1:]), "'", '').astype(np.float)
#real_data_small = [np.concatenate(([item[0]], item[-13:-11], item[-5:-2], [item[-1]])) for item in real_data]
real_data_small = real_data
real_data_small_no_class = [item2[:-1] for item2 in real_data_small]
answers = [int(item3[-1]) for item3 in real_data_small]
print("Answers: ", answers)

print(real_data_small_no_class[0:10])

# swiss_roll = make_swiss_roll(n_samples=500, noise=0.1, random_state=None)[0]

# iris = datasets.load_iris()
# iris_data = iris.data[:, :3]
# END OF DATASETS -----------------------------------------------------------------------------------------------------

raw_data = big_small_blob
dims = len(raw_data[0])
random_data = (np.random.rand(100, dims) - 0.5) * 40
data = np.concatenate((big_small_blob, random_data))

count_anomaly = 0
for item in real_data_small:
    if item[-1] > 0.5:
        count_anomaly += 1

#print("part of anomalies in dataset: ", count_anomaly / len(real_data_small), "number of anomalies: ", count_anomaly)
print("anomalies in dataset: ", answers_shuttle.count(-1), "percentage: ", answers_shuttle.count(-1)/len(answers_shuttle))

def main(plot_it=False):
    for j in ct:
        st = time.time()
        results_if = IsolationForest(contamination=j, n_estimators=100).fit_predict(shuttle_data_no_class)
        #for index, res_if in enumerate(results_if):
        #    if res_if == -1:
        #        results_if[index] = 1
        #    else:
        #        results_if[index] = -1
        print(results_if)
        ctr = 0
        ctr2 = 0
        for i, res in enumerate(results_if):
            if answers_shuttle[i] == -1 and res == -1:
                ctr += 1
            if res == -1:
                ctr2 += 1
        print("Contamination: ", j)
        print("True Positives: ", ctr, "percentage correct: ", ctr / int(j * len(answers_shuttle)))
        print("Total Positives: ", ctr2)
        print("ROC-AUC Score: ", roc_auc_score(answers_shuttle, results_if))
        print("Running time: ", time.time()-st)
        #print("prediction: ", results_if[-100:])
        #print("answers: ", answers[-100:])
    start_time = time.time()
    rdf_points = shuttle_data.values.tolist()
    rdf = []
    for point in rdf_points:
        pt = []
        for value in point:
            if value > 100:
                value = 99
            if value < -100:
                value = -99
            pt.append(value)
        rdf.append(point)
    print("max in rdf: ", np.amax(rdf))
    clf = RDForest(contamination=ct2, k=10, points=rdf)
    result = clf.fit_predict()
    results = [p.is_outlier for p in result]
    filtered_results = [p for p in result if p.is_outlier == -1]
    filtered_results = sorted(filtered_results[:int(len(result) * ct2)], key=lambda x: x.anomaly_score)
    table = PrettyTable((['Coordinates', 'Anomaly Score']))
    for res in filtered_results:
        table.add_row([res.coordinates[-1], res.anomaly_score])
    print(table)
    c = 0
    for it in filtered_results:
        if it.coordinates[-1] not in [1, 4]:
            c += 1
    print("percentage of anomalies in final results: ", c / len(filtered_results), "anomalies found: ",
          c, "normal points found: ", len(filtered_results) - c)
    print("ROC-AUC score: ", roc_auc_score(answers_shuttle, results))
    print("Answers: ", answers_shuttle[-100:])
    print("Results: ", results[-100:])
    print("--- running time: {:.2f} seconds ---".format(time.time() - start_time))
    if plot_it:
        tdplot(result)

    # base = clf.trees[0].root
    # dt = build_tree(base)
    # dt.render()
    # print("rendered")


main(plot_it=False)
