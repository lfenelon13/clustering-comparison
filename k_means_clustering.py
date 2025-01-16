from sklearn.cluster import KMeans
import numpy as np

# Sample data
data = np.array([[1, 2], [1, 4], [1, 0],
                 [10, 2], [10, 4], [10, 0]])

# Create KMeans instance
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)

# Get cluster labels
labels = kmeans.labels_
print(labels)

#hello
