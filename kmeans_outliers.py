import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
X, label = make_blobs(100, centers = 1)

# Z-score
from scipy.stats import zscore
import pandas as pd
zscore_X  = zscore(X)

df = pd.DataFrame(zscore_X, columns =['X', 'X_zscore'])
df["is_outlier"]  = df['X_zscore'].apply (
 lambda x: x <= -1.5 or x >= 1.5 )

print (df[df["is_outlier"]])


# DBSCAN
from sklearn.cluster import DBSCAN
outlier_detection = DBSCAN(
  eps = 0.5,
  metric="euclidean",
  min_samples = 3,
  n_jobs = -1)
clusters = outlier_detection.fit_predict(X)

print (clusters)

from matplotlib import cm
cmap = cm.get_cmap('Accent')
df.plot.scatter(
  x = "X",
  y = "X_zscore",
  c = clusters,
  cmap = cmap,
  colorbar = False
)

# K-Means

import numpy as np
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=1)

kmeans.fit(X)

f, ax = plt.subplots(figsize=(7,5))
ax.set_title('Blob')
ax.scatter(X[:, 0], X[:, 1], label='Points')
ax.scatter(kmeans.cluster_centers_[:, 0],
           kmeans.cluster_centers_[:, 1], label='Centroid',
           color='r')
ax.legend(loc='best')

plt.show()


distances = kmeans.transform(X)

sorted_idx = np.argsort(distances.ravel())[::-1][:5]

f, ax = plt.subplots(figsize=(7,5))
ax.set_title('Single Cluster')
ax.scatter(X[:, 0], X[:, 1], label='Points')
ax.scatter(kmeans.cluster_centers_[:, 0],
           kmeans.cluster_centers_[:, 1],
           label='Centroid', color='r')
ax.scatter(X[sorted_idx][:, 0],
           X[sorted_idx][:, 1],
           label='Extreme Value', edgecolors='g',
           facecolors='none', s=100)
ax.legend(loc='best')

plt.show()

# simulating removing these outliers
new_X = np.delete(X, sorted_idx, axis=0)

# this causes the centroids to move slightly
new_kmeans = KMeans(n_clusters=1)
new_kmeans.fit(new_X)


f, ax = plt.subplots(figsize=(7,5))
ax.set_title("Extreme Values Removed")
ax.scatter(new_X[:, 0], new_X[:, 1], label='Pruned Points')
ax.scatter(kmeans.cluster_centers_[:, 0],
           kmeans.cluster_centers_[:, 1],
           label='Old Centroid',
           color='r', s=80, alpha=.5)
ax.scatter(new_kmeans.cluster_centers_[:, 0],
           new_kmeans.cluster_centers_[:, 1],
           label='New Centroid',
           color='m', s=80, alpha=.5)
ax.legend(loc='best')

from scipy import stats
emp_dist = stats.multivariate_normal(kmeans.cluster_centers_.ravel())
lowest_prob_idx = np.argsort(emp_dist.pdf(X))[:5]
np.all(X[sorted_idx] == X[lowest_prob_idx])



print ( kmeans.cluster_centers_)
print (kmeans.cluster_centers_.ravel())