# K-Means-Clustering-Project
K-Means clustering is the most popular unsupervised machine learning algorithm.
K-Means clustering is used to find intrinsic groups within the unlabelled dataset and draw inferences from them. I have used Facebook Live Sellers in Thailand dataset for this project. I implement K-Means clustering to find intrinsic groups within this dataset that display the same status_type behaviour. The status_type behaviour variable consists of posts of a different nature (video, photos, statuses and links).

**Table of Contents**
Introduction to K-Means Clustering

K-Means Clustering intuition

Choosing the value of K
The elbow method
The problem statement
Dataset description
Import libraries
Import dataset
Exploratory data analysis
Declare feature vector and target variable
Convert categorical variable into integers
Feature scaling
K-Means model with two clusters
K-Means model parameters study
Check quality of weak classification by the model
Use elbow method to find optimal number of clusters
K-Means model with different clusters
Results and conclusion

**1. Introduction to K-Means Clustering**
Machine learning algorithms can be broadly classified into two categories - supervised and unsupervised learning. There are other categories also like semi-supervised learning and reinforcement learning. But, most of the algorithms are classified as supervised or unsupervised learning. The difference between them happens because of presence of target variable. In unsupervised learning, there is no target variable. The dataset only has input variables which describe the data. This is called unsupervised learning.

**K-Means clustering** is the most popular unsupervised learning algorithm. It is used when we have unlabelled data which is data without defined categories or groups. The algorithm follows an easy or simple way to classify a given data set through a certain number of clusters, fixed apriori. K-Means algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity.

**2. K-Means Clustering intuition**
K-Means clustering is used to find intrinsic groups within the unlabelled dataset and draw inferences from them. It is based on centroid-based clustering.

Centroid - A centroid is a data point at the centre of a cluster. In centroid-based clustering, clusters are represented by a centroid. It is an iterative algorithm in which the notion of similarity is derived by how close a data point is to the centroid of the cluster. K-Means clustering works as follows:- The K-Means clustering algorithm uses an iterative procedure to deliver a final result. The algorithm requires number of clusters K and the data set as input. The data set is a collection of features for each data point. The algorithm starts with initial estimates for the K centroids. The algorithm then iterates between two steps:-

**1. Data assignment step**

Each centroid defines one of the clusters. In this step, each data point is assigned to its nearest centroid, which is based on the squared Euclidean distance. So, if ci is the collection of centroids in set C, then each data point is assigned to a cluster based on minimum Euclidean distance.

**2. Centroid update step**

In this step, the centroids are recomputed and updated. This is done by taking the mean of all data points assigned to that centroid’s cluster.

The algorithm then iterates between step 1 and step 2 until a stopping criteria is met. Stopping criteria means no data points change the clusters, the sum of the distances is minimized or some maximum number of iterations is reached. This algorithm is guaranteed to converge to a result. The result may be a local optimum meaning that assessing more than one run of the algorithm with randomized starting centroids may give a better outcome.

**3. Choosing the value of K**
The K-Means algorithm depends upon finding the number of clusters and data labels for a pre-defined value of K. To find the number of clusters in the data, we need to run the K-Means clustering algorithm for different values of K and compare the results. So, the performance of K-Means algorithm depends upon the value of K. We should choose the optimal value of K that gives us best performance. There are different techniques available to find the optimal value of K. The most common technique is the elbow method which is described below.

**4. The elbow method**
The elbow method is used to determine the optimal number of clusters in K-means clustering. The elbow method plots the value of the cost function produced by different values of K.

If K increases, average distortion will decrease. Then each cluster will have fewer constituent instances, and the instances will be closer to their respective centroids. However, the improvements in average distortion will decline as K increases. The value of K at which improvement in distortion declines the most is called the elbow, at which we should stop dividing the data into further clusters.

**5. The problem statement**
In this project, I implement K-Means clustering with Python and Scikit-Learn. As mentioned earlier, K-Means clustering is used to find intrinsic groups within the unlabelled dataset and draw inferences from them. I have used Facebook Live Sellers in Thailand Dataset for this project. I implement K-Means clustering to find intrinsic groups within this dataset that display the same status_type behaviour. The status_type behaviour variable consists of posts of a different nature (video, photos, statuses and links).

**6. Dataset description**
In this project, I have used Facebook Live Sellers in Thailand Dataset, downloaded from the UCI Machine Learning repository. The dataset can be found at the following url-

https://archive.ics.uci.edu/ml/datasets/Facebook+Live+Sellers+in+Thailand

The dataset consists of Facebook pages of 10 Thai fashion and cosmetics retail sellers. The status_type behaviour variable consists of posts of a different nature (video, photos, statuses and links). It also contains engagement metrics of comments, shares and reactions.
