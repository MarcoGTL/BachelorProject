from sklearn.cluster import KMeans, SpectralClustering


def k_means(data, num_clusters):
    clustering = KMeans(n_clusters=num_clusters).fit(data)
    return clustering.labels_


def spectral(data, num_clusters):
    clustering = SpectralClustering(n_clusters=num_clusters).fit(data)
    return clustering.labels_
