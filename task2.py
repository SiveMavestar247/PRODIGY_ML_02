import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Mall_Customers.csv')

# Selecting the features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # You can change the number of clusters
kmeans.fit(X)

# Add the cluster labels to the original data
data['Cluster'] = kmeans.labels_

# Visualization of the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.title('Customer Segments')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

# Analyze the results
print(data.head())