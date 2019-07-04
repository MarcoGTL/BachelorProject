from sklearn.datasets import load_digits
from sklearn.manifold import MDS

"""
Author: Marco

A class performing mds to transform data to two dimensional data. Used for plotting feaature vector

Required Packages:
    scikit-learn

    Attributes:
        data: List of lists               # In this case the feature vector of our pipeline
"""
def mds_transform(data):
    embedding = MDS(n_components=2)
    data_transformed =  embedding.fit_transform(data)
    return data_transformed


