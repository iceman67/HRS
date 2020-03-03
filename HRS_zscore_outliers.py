
#
# Z값(Z - value), Z 점수(Zscore) s a numerical measurement used in statistics of a value's relationship
# to the mean (average) of a group of values measured in terms of standard deviations from the mean.
# how far from the mean a data point is
# z = (x – μ) / σ
# mean (μ),  standard deviation (σ)

# Feb. 25 2020 Yunh. Kang
# add --input

import pandas as pd
import numpy as np
from scipy.stats import zscore

import argparse


def parse_args():
 # Parse command line arguments
 ap = argparse.ArgumentParser(description="Glove HighTemp data receiver")
 ap.add_argument("-i", "--input", required=True,
                 help="input csv")
 return vars(ap.parse_args())


# find outliers  g4 based on z-score
def find_outlier_g4(df):
    # construct 1d array of observation column
    observation = df['observation'].to_numpy()
    observation = observation.reshape(1, -1)

    # construct 1d array of reference column
    reference = df['reference'].to_numpy()
    reference = reference.reshape(1, -1)


    group4 = df.count()[0]//4

    new_df = pd.DataFrame(columns=['observation', 'reference'])

    for i in range (1,group4):
        group4_ob_mean = df.iloc[0*i:4*i]['observation'].mean()
        group4_ref_mean = df.iloc[0*i:4*i]['reference'].mean()

        new_df.loc[i] = [group4_ob_mean, group4_ref_mean]


    observation = new_df['observation'].to_numpy()
    observation = observation.reshape(1, -1)

    reference = new_df['reference'].to_numpy()
    reference = reference.reshape(1, -1)

    X = np.append(observation, reference)
    X = np.reshape(X, (-1, 2))

    zscore_X = zscore(X)

    df = pd.DataFrame(zscore_X, columns=['Ob', 'Re'])

    df["is_outlier"] = df['Ob'].apply(
        lambda x: x <= -1.5 or x >= 1.5)
    print(df[df["is_outlier"]])

    num = np.sum(df["is_outlier"] == True)
    rate = num / len(X[:, 1])
    print("# of outliers = {}, {}% ".format(num, rate * 100))

    df["is_outlier"] = df['Re'].apply(
        lambda x: x <= -1.5 or x >= 1.5)

    print(df[df["is_outlier"]])

    # print (df[df["is_outlier"]])
    num = np.sum(df["is_outlier"] == True)
    rate = num / len(X[:, 1])
    print("# of outliers = {}, {}% ".format(num, rate * 100))



def print_outlier(df, col_name, X):
    df["is_outlier"] = df[col_name].apply(
        lambda x: x <= -1.5 or x >= 1.5)

    print(df[df["is_outlier"]])

    # print (df[df["is_outlier"]])
    num = np.sum(df["is_outlier"] == True)
    rate = num / len(X[:, 1])
    print("# of outliers = {}, {}% ".format(num, rate * 100))


def find_outlier_c4(df):
    # construct 1d array of observation column
    observation = df['observation'].to_numpy()
    observation = observation.reshape(1, -1)

    # construct 1d array of reference column
    reference = df['reference'].to_numpy()
    reference = reference.reshape(1, -1)


    group4 = df.count()[0]//4

    new_df = pd.DataFrame(columns=['ob1', 'ob2', 'ob3', 'ob4'])

    for i in range (1,group4):
        ob1 = df.iloc[0 * i]['observation']
        ob2 = df.iloc[1 * i]['observation']
        ob3 = df.iloc[2 * i]['observation']
        ob4 = df.iloc[3 * i]['observation']
        new_df.loc[i] = [ob1, ob2, ob3, ob4]


    nd_ob1 = new_df['ob1'].to_numpy()
    nd_ob1 = nd_ob1.reshape(1, -1)
    nd_ob2 = new_df['ob2'].to_numpy()
    nd_ob2 = nd_ob2.reshape(1, -1)
    nd_ob3 = new_df['ob3'].to_numpy()
    nd_ob3 = nd_ob3.reshape(1, -1)
    nd_ob4 = new_df['ob4'].to_numpy()
    nd_ob4 = nd_ob4.reshape(1, -1)


    arr1 = np.append(nd_ob1, nd_ob2)
    arr2 = np.append(arr1, nd_ob3)
    X = np.append(arr2, nd_ob4)
    X = np.reshape(X, (-1, 4))
    #print(X)

    zscore_X = zscore(X)
    print (zscore_X.shape)

    df = pd.DataFrame(zscore_X, columns=['X1', 'X2', 'X3', 'X4'])

    print_outlier(df, 'X1', X)
    print_outlier(df, 'X2', X)
    print_outlier(df, 'X3', X)
    print_outlier(df, 'X4', X)

    '''
    df["is_outlier"] = df['X1'].apply(
        lambda x: x <= -1.5 or x >= 1.5)

    print(df[df["is_outlier"]])




    # print (df[df["is_outlier"]])
    num = np.sum(df["is_outlier"] == True)
    rate = num / len(X[:, 1])
    print("# of outliers = {}, {} ".format(num, rate * 100))

    '''

# find outliers based on z-score
def find_outlier(df):
    # construct 1d array of observation column
    observation = df['observation'].to_numpy()
    observation = observation.reshape(1, -1)

    '''
    group4 = df.count()[0]//4
    new_df = pd.DataFrame(columns=['observation4'])

    for i in range (1,group4):
        group4_mean = df.iloc[0*i:4*i]['observation'].mean()
        new_df.loc[i] = [group4_mean]

    print(new_df.head(3))
    '''



    # construct 1d array of reference column
    reference = df['reference'].to_numpy()
    reference = reference.reshape(1, -1)

    # construct 2d array of reference column
    X = np.append(observation, reference)
    X = np.reshape(X, (-1, 2))

    # (1001, 2)

    numX = X.shape[0]
    print(X.shape[0])
    # print (X)

    zscore_X  = zscore(X)
    df = pd.DataFrame(zscore_X, columns =['Ob', 'Re'])

    df["is_outlier"]  = df['Ob'].apply (
    lambda x: x <= -1.5 or x >= 1.5 )
    print(df[df["is_outlier"]])

    num = np.sum(df["is_outlier"] == True)
    rate = num / len(X[:, 1])
    print("# of outliers = {}, {}% ".format(num, rate * 100))

    df["is_outlier"]  = df['Re'].apply (
    lambda x: x <= -1.5 or x >= 1.5 )

    print(df[df["is_outlier"]])

    # print (df[df["is_outlier"]])
    num =  np.sum(df["is_outlier"] == True)
    rate = num/len(X[:,1])
    print ("# of outliers = {}, {}% ".format(num, rate*100))



def main(args):
    df = pd.read_csv(args["input"])

    #print("GROUP 4 by mean - observation vs reference[KF]")
    #find_outlier_g4(df)
    #print("\n4 SENSORS  without reference")
    #find_outlier_c4(df)
    print("\nWITHOUT grouping [4 sensors] vs reference")
    find_outlier(df)


if __name__ == '__main__':
    args = parse_args()
    main(args)


'''

# construct 1d array of observation column
observation = df['observation'].to_numpy()
observation = observation.reshape(1, -1)

# construct 1d array of reference column
reference = df['reference'].to_numpy()
reference = reference.reshape(1, -1)

# construct 2d array of reference column
X = np.append(observation, reference)
X = np.reshape(X, (-1, 2))
# (1001, 2)
print(X.shape)
#print (X)


zscore_X  = zscore(X)
df = pd.DataFrame(zscore_X, columns =['X', 'X_zscore'])

df["is_outlier"]  = df['X_zscore'].apply (
 lambda x: x <= -1.5 or x >= 1.5 )
print (df[df["is_outlier"]])


num =  np.sum(df["is_outlier"] == True)
rate = num/len(X[:,1])

print ("# of outliers = {}, {} ".format(num, rate*100))
'''

'''
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
'''
