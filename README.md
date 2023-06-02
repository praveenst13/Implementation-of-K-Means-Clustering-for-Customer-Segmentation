# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Choose the number of clusters (K): 
          Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

2.Initialize cluster centroids: 
        Randomly select K data points from your dataset as the initial centroids of the clusters.

3.Assign data points to clusters: 
      Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically  done using Euclidean distance, but other distance metrics can also be used.

4.Update cluster centroids: 
      Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

5.Repeat steps 3 and 4: 
      Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

6.Evaluate the clustering results: 
      Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

 7.Select the best clustering solution: 
      If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Praveen s
RegisterNumber:  212222240077
*/
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("/content/Dataset-20230524.zip")
data

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])

y_pred = km.predict(data.iloc[:,3:])
y_pred

data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="yellow",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="pink",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="purple",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:
### Data Head
![O1](https://github.com/LATHIKESHWARAN/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393556/6657f71f-0ad3-421e-aa14-3e21fe790722)
### Data info
![O2](https://github.com/LATHIKESHWARAN/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393556/c51ff8b1-92fb-4b52-8cf7-b5b371110f0b)
### Null Values
![O3](https://github.com/LATHIKESHWARAN/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393556/afefce26-8ae7-4d60-b051-bf4b588315a2)
### Elbow Graph
![O4](https://github.com/LATHIKESHWARAN/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393556/6548cfdd-faa1-47e5-b3fc-c283485e3160)
### Cluster Formation
![O5](https://github.com/LATHIKESHWARAN/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393556/9e103008-762f-4562-89de-79ac2a78c3fd)
### Predicted Value
![O6](https://github.com/LATHIKESHWARAN/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393556/8f96b716-ddab-4413-9a3c-3f37c3932054)
### Final Graph
![O7](https://github.com/LATHIKESHWARAN/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393556/674cb910-96a0-4170-88d7-946a8117892f)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
