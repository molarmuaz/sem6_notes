# Clustering
Clustering is an unsupervised machine learning model. The objective is to find clusters among data without explicitly having to mention the features that might cause two data points to be in the same cluster. Since we aren't using any labels or defining the basis upon which we need clusters, this is not supervised machine learning. For example I want to get groups of audience demographics I have. I might not know on what factor they relate. A clustering algorithm can find common patterns among them to separate them into groups

## Major Approaches
- **Exclusive**
	- One data can only be part of one cluster. e.g. K-means.
- **Agglomerative**
	- All datapoints start of as a cluster and through iterative convergence we reduce the number of clusters by joining nearest clusters. e.g. Hierarchical Clustering.
- **Overlapping**
	- We use fuzzy sets to cluster data, which means a single datapoint can be part of multiple clusters with different degrees of membership value. This means a points can be partially part of multiple clusters to a certain degree. for example, "He is 0.7 young", "He is 0.4 tall". Fuzzy C-Means is a model that practices this.
- **Probabilistic**
	- Uses probability distribution measures to create clusters. Similar to overlapping clustering, we can have a datapoint be in multiple clusters, but instead of saying we know this data belongs in this cluster but by how much, we say how likely is this to be in this cluster. For example "this customer is 60% likely to be a frequent customer". Gaussian Mixture Model is a model that uses this clustering approach.

### Exclusive
Divide N objects into K set of clusters.
#### K-means
Given the value of K (number of clusters), K-means is implemented in 4 steps:
1. Select K points at random as cluster centers (centroids)
2. Assign each instance to its closest cluster center using certain distance measure (usually Euclidean or Manhattan)
3. Calculate the centroid of each cluster, use it as the new cluster center (one measure of centroid is mean)
4. Go back to Step 2, stop when cluster centers do not change anymore

###### Distance Formulas

$$
\text{Euclidean}(A, B) = \sqrt{ \sum_{i=1}^{n} (x_i - y_i)^2 }
$$
$$
\text{Manhattan}(A, B) = \sum_{i=1}^{n} |x_i - y_i|
$$
#### Issues
- Does not necessarily find the most optimal configuration
- Results depend a lot on the initial randomly selected cluster centers
- Number of clusters need to be specified where we don't even know on what basis the clusters are going to get made.

### Agglomerative
This clustering does not require the number of clusters to be set. After converging iteratively, we end up with one big cluster that contains all the objects. But, what's the use of having one cluster?

#### Dendrogram
While merging cluster one by one, we can draw a tree diagram known as dendrogram. From this diagram we can get however many clusters we want.

!(attachment/Clustering/file-.jpg)

This is what it looks like, so if we want two clusters:
– Cluster 1: q, r
– Cluster 2: x, y, z, p

and if we want three clusters:
– Cluster 1: q, r
– Cluster 2: x, y, z
– Cluster 3: p

#### Hierarchical 
1. Start by assigning each item to its own cluster, so that if you have N items, you now have N clusters, each containing just one item
2. Find the closest distance (most similar) pair of clusters and merge them into a single cluster, so that now you have one less cluster
3. Compute pairwise distances between the new cluster and each of the old clusters
4. Repeat steps 2 and 3 until all items are clustered into a single cluster of size N
5. Draw the dendrogram, and with the complete hierarchical tree, if you want K clusters you just have to cut the K-1 top links

Computing distances between clusters for Step 3 can be implemented in different ways:
-  **Single-linkage clustering**
	-  The distance between one cluster and another cluster is computed as the shortest distance from any member of one cluster to any member of the other cluster
-  **Complete-linkage clustering**
	-  The distance between one cluster and another cluster is computed as the greatest distance from any member of one cluster to any member of the other cluster
-  **Centroid clustering**
	-  The distance between one cluster and another cluster is computed as the distance from one cluster centroid to the other cluster centroid

!(attachment/Clustering/file- 1.jpg)

### How to know we have good clusters?

In general, **good clusters** should have:

- **High intra-cluster similarity**,  
    i.e. **low variance** among the members of the same cluster.
  
$$
\text{Var}(x) = \frac{1}{N-1} \sum_{i=1}^{N} (x_i - \bar{x})^2
$$
However, only having low intra-cluster variance doesn't necessarily mean it is good clustering, we also need there to be high inter-cluster variance i.e. the clusters themselves are far from each other while the cluster itself should be tight. We have a metric that evaluates both inter and intra cluster variance together. This is known as Davies-Bouldin Index.

#### Davies-Bouldin Index
Let \( C = \{C_1, C_2, \dots, C_k\} \) be the set of clusters.

**Davies-Bouldin Index:**

$$
\text{DB} = \frac{1}{k} \sum_{i=1}^{k} R_i
$$

where

$$
R_i = \max_{j \ne i} R_{ij}
$$

and

$$
R_{ij} = \frac{\text{var}(C_i) + \text{var}(C_j)}{\|c_i - c_j\|}
$$

The lower the result, the better the clustering

###### Example
Cluster 1 :  3
Cluster 2:   7, 10
Cluster 3:   17, 18, 20

var(C1) = 0
var(C2) = (7-8.5)^2 + (10-8.5)^2 = 4.5
var(C3) = 2.33

R12 = 0 + 4.5/ |3-8.5| = 4.5/5.5  = 0.8181
R13 = 0 + 2.33 / |3-18.33| = 0.1519
R23 = 4.5 + 2.33 / |8.5-18.33| = 0.6948

R1 = max(R12, R13) = 0.8181
R2 = max(R12, R23) = 0.8181
R3 = max(R13, R23) = 0.6948

DB = 1/3 * (0.8181+0.8181+0.6948) = 0.777
