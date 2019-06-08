from sklearn.datasets import load_digits
from sklearn.manifold import MDS
from matplotlib import pyplot

def mds_transform(data):
    embedding = MDS(n_components=2)
    data_transformed =  embedding.fit_transform(data)
    return data_transformed


if __name__ == '__main__':
    X, _ = load_digits(return_X_y=True)
    print(X)
    print(X.shape)
    embedding = MDS(n_components=2)
    X_transformed = embedding.fit_transform(X[:100])
    print(X_transformed.shape)


