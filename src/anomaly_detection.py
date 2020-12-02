# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Albert Thomas <albert.thomas@telecom-paristech.fr>
# License: BSD 3 clause

import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from src.rsqt_forest import RSQT

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

# Example settings
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers
acc_res = []  # for the accuracy results

# define outlier/anomaly detection methods to be compared
anomaly_algorithms = [
    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1)),
    ("Isolation Forest", IsolationForest(contamination=outliers_fraction,
                                         random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(
        n_neighbors=35, contamination=outliers_fraction)),
    ("RSQT Forest", RSQT(contamination=outliers_fraction))]

# generate custom datasets
lines = []
dom = 3
for i in range(int(n_inliers / 2)):
    lines.append([dom * np.random.randn(), dom])
    lines.append([dom * np.random.randn(), -dom])

rectangles = []
for i in range(int(n_inliers/12)):
    rectangles.append([4. * (np.random.random() - 0.5), 2])
    rectangles.append([4. * (np.random.random() - 0.5), -2])
    rectangles.append([2, 4. * (np.random.random() - 0.5)])
    rectangles.append([-2, 4. * (np.random.random() - 0.5)])
for i in range(int(n_inliers / 6)):
    rectangles.append([8. * (np.random.random() - 0.5), 4])
    rectangles.append([8. * (np.random.random() - 0.5), -4])
    rectangles.append([4, 8. * (np.random.random() - 0.5)])
    rectangles.append([-4, 8. * (np.random.random() - 0.5)])

plus = []
for i in range(int(n_inliers/2)+1):
    plus.append([10. * (np.random.random() - 0.5), 0.1 * np.random.randn()])
    plus.append([0.1 * np.random.randn(), 10. * (np.random.random() - 0.5)])

t_sign = []
for i in range(int(n_inliers/2)):
    t_sign.append([8. * (np.random.random() - 0.5), 4 + 0.1 * np.random.randn()])
    t_sign.append([0.1 * np.random.randn(), 8. * (np.random.random() - 0.5)])

big_small_blob = make_blobs(centers=[[0, 0], [5, 5]], n_samples=[int(n_inliers * 0.8), int(n_inliers * 0.2)],
                            n_features=2, cluster_std=[2., .2])[0]

# Define datasets
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
datasets = [
    np.concatenate((6. * make_circles(n_samples=int(n_inliers / 2), factor=0.8)[0],
                    3. * make_circles(n_samples=int(n_inliers / 2),
                                      factor=0.5)[0])),
    rectangles,
    big_small_blob,
    plus,
    t_sign
    # make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
    # **blobs_params)[0],
    # 5. * make_circles(n_samples=n_inliers, factor=0.6)[0],
    # make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],
    # **blobs_params)[0],
    # make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
    # **blobs_params)[0],
    # 4. * (make_moons(n_samples=n_inliers, noise=.05, random_state=0)[0] -
    # np.array([0.5, 0.25])),
    # 14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)
    # lines]
]

# Compare given classifiers under given settings
xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                     np.linspace(-7, 7, 150))

plt.figure(figsize=(len(anomaly_algorithms) * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1
rng = np.random.RandomState(42)

ss = time.time()

for i_dataset, X in enumerate(datasets):
    # Add outliers
    X = np.concatenate([X, rng.uniform(low=-6, high=6,
                                       size=(n_outliers, 2))], axis=0)
    c = []
    t = []
    count = 0
    correct = 0

    for name, algorithm in anomaly_algorithms:
        t0 = time.time()
        points = []
        if name != "RSQT Forest":
            algorithm.fit(X)
        t1 = time.time()
        plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        # fit the data and tag outliers
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X)
        elif name == "RSQT Forest":
            points, y_pred = algorithm.fit_predict(X)
            t = y_pred
        else:
            y_pred = algorithm.fit(X).predict(X)
            if name == "Isolation Forest":
                c = y_pred
        # plot the levels lines and the points
        if name not in ["Local Outlier Factor", "RSQT Forest"]:  # LOF does not implement predict
            Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

        colors = np.array(['#377eb8', '#ff7f00'])

        if name != "RSQT Forest":
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])
        else:
            plt.scatter([p.x for p in points], [p.y for p in points], s=10,
                        color=colors[[y_pr for y_pr in y_pred]])
        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

    # calculate percentage of correct predictions
    for i in range(len(c)):
        if c[i] == -1:
            count += 1
            if t[i] == 0:
                correct += 1
    acc_res.append(100 * correct / count)

# plt.savefig('../output/methods_comparison4.pdf')

for i in range(len(datasets)):
    print("percentage of same predictions for dataset {}: {:.2f} ".format(i + 1, acc_res[i]), "%")
print("running time: {:.2f}".format(time.time() - ss), "s")

plt.show()
