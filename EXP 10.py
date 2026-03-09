# Import required libraries
import numpy as np
from sklearn.mixture import GaussianMixture

# Sample dataset
X = np.array([[1], [2], [3], [10], [11], [12]])

# Create EM model using Gaussian Mixture
gmm = GaussianMixture(n_components=2)

# Train the model
gmm.fit(X)

# Predict cluster labels
labels = gmm.predict(X)

# Print results
print("Data Points:", X.flatten())
print("Cluster Labels:", labels)
print("Means:", gmm.means_)
print("Weights:", gmm.weights_)
