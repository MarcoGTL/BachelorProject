import cv2
import numpy as np
import slic
import clustering
import features
import histograms
from os import listdir, path
import maxflow
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from scipy.spatial.distance import euclidean


class Algorithms:
    def __init__(self, image_paths):
        self.images = image_paths                                # Relative paths to the images to be used
        # Each dictionary stores the data for each image using its image path as a key
        self.imgs_bgr = dict()                               # Float64 representation of the image
        self.imgs_segmentation = dict()                          # Stores for each pixel its superpixel it belongs to
        self.imgs_segment_ids = dict()                           # List of superpixel indices
        self.imgs_segment_neighbors = dict()                     # Stores for each superpixel a list of its neighbors
        self.imgs_segment_centers = dict()
        self.imgs_segment_feature_vectors = dict()
        self.imgs_foreground_segments = dict.fromkeys(image_paths, [])  # List of foreground superpixels
        self.imgs_background_segments = dict.fromkeys(image_paths, [])  # List of background superpixels
        self.gmm_fg = None
        self.gmm_bg = None
        self.gmm_fg_bic = 0
        self.gmm_bg_bic = 0
        self.imgs_uncertainties_node = dict()
        self.imgs_uncertainties_edge = dict()
        self.imgs_uncertainties_graph_cut = dict()
        self.imgs_cosegmented = dict()            # Stores for each pixel its segment it belongs to after cosegmentation

    # generate superpixel segments for all images using SLIC
    def compute_superpixels_slic(self, num_segments, compactness=10.0, max_iter=10, sigma=0):
        for img in self.images:
            self.imgs_bgr[img] = cv2.imread(img, flags=cv2.IMREAD_COLOR)
            self.imgs_segmentation[img] = slic.get_segmented_image(self.imgs_bgr[img], num_segments,
                                                                   compactness, max_iter, sigma)
            self.imgs_segment_ids[img] = np.unique(self.imgs_segmentation[img])

    # compute the neighbor superpixels for each superpixel
    def compute_neighbors(self):
        for img in self.images:
            self.imgs_segment_neighbors[img] = [set() for i in self.imgs_segment_ids[img]]
            for row in range(len(self.imgs_segmentation[img])-1):
                for column in range(len(self.imgs_segmentation[img][0])-1):
                    current = self.imgs_segmentation[img][row][column]  # superpixel of current pixel
                    right = self.imgs_segmentation[img][row][column+1]  # superpixel of pixel right of current
                    below = self.imgs_segmentation[img][row+1][column]  # superpixel of pixel below current
                    if current != right:
                        self.imgs_segment_neighbors[img][current].add(right)
                        self.imgs_segment_neighbors[img][right].add(current)
                    if current != below:
                        self.imgs_segment_neighbors[img][current].add(below)
                        self.imgs_segment_neighbors[img][below].add(current)

    # compute the center of each superpixel
    # this method may yield locations outside superpixels with odd shapes
    def compute_centers(self):
        for img in self.images:
            self.imgs_segment_centers[img] = []
            for i in self.imgs_segment_ids[img]:
                # Retrieve indices of segment i
                indices = np.where(self.imgs_segmentation[img] == i)
                # Center is mean of indices
                self.imgs_segment_centers[img].append((np.median(indices[1]), np.median(indices[0])))

    # sets the foreground of the image at image_path to segments
    def set_fg_segments(self, image_path, segments):
        self.imgs_foreground_segments[image_path] = segments

    # sets the background of the image at image_path to segments
    def set_bg_segments(self, image_path, segments):
        self.imgs_background_segments[image_path] = segments

    # By default calculates a feature vector consisting of means of B, G, R, H, S, V,
    # and a histogram and their entropy of Hue (5 bins) and Saturation (3bins)
    def compute_feature_vectors(self, means_bgr=True, means_hsv=True,
                                h_hist=True, h_hist_bins=5, h_hist_entropy=True,
                                s_hist=True, s_hist_bins=3, s_hist_entropy=True,
                                hs_hist=False, hs_hist_bins_h=5, hs_hist_bins_s=3,
                                sift=False, sift_kp_size=32.0,
                                hog=False, hog_winSize=(32, 32), hog_blockSize=(16, 16), hog_blockStride=(8, 8),
                                hog_cellSize=(8, 8), hog_bins=9):
        for img in self.images:
            self.imgs_segment_feature_vectors[img] = [0 for i in self.imgs_segment_ids[img]]
            image_bgr = self.imgs_bgr[img]
            image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            for segment in self.imgs_segment_ids[img]:
                feature_vector = []
                mask = np.uint8(self.imgs_segmentation[img] == segment)
                coord_xy = (self.imgs_segment_centers[img][segment][0], self.imgs_segment_centers[img][segment][1])
                if means_bgr:
                    feature_vector += features.get_means_bgr_fv(image_bgr, mask)
                if means_hsv:
                    feature_vector += features.get_means_hsv_fv(image_hsv, mask)
                if h_hist:
                    feature_vector += features.get_h_hist_fv(image_hsv, mask, h_hist_bins, h_hist_entropy)
                if s_hist:
                    feature_vector += features.get_s_hist_fv(image_hsv, mask, s_hist_bins, s_hist_entropy)
                if hs_hist:
                    feature_vector += features.get_normalized_hs_hist_fv(image_hsv, mask, hs_hist_bins_h, hs_hist_bins_s)
                if sift:
                    feature_vector += features.get_sift_fv(image_gray, coord_xy[0], coord_xy[1], sift_kp_size)
                if hog:
                    feature_vector += features.get_hog_fv(image_bgr, coord_xy[0], coord_xy[1], hog_winSize, hog_blockSize, hog_blockStride, hog_cellSize, hog_bins)
                assert len(feature_vector) > 0, 'Feature vector needs to have at least one feature'
                self.imgs_segment_feature_vectors[img][segment] = np.asarray(feature_vector, dtype='float32')

    # Fits several Gaussian Mixture Model to the foreground and background superpixels and chooses the best fit
    # components_range : try to fit a model for these number of components
    # n_init : attempts per number of components
    def compute_gmm(self, components_range=range(5, 9), n_init=10):
        # group the foreground and background segments' feature vectors in one list
        feature_vectors_fg = [self.imgs_segment_feature_vectors[img][fg_segment] for img in self.images for fg_segment
                              in self.imgs_foreground_segments[img]]
        feature_vectors_bg = [self.imgs_segment_feature_vectors[img][bg_segment] for img in self.images for bg_segment
                              in self.imgs_background_segments[img]]

        def find_best_gmm(X):
            lowest_bic = np.infty
            for n_components in components_range:
                # Fit a Gaussian mixture with EM
                gmm = GaussianMixture(n_components=n_components, covariance_type='full', n_init=n_init)
                gmm.fit(X)
                bic = gmm.bic(np.asarray(X, dtype=np.float32))
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
            return best_gmm, lowest_bic

        self.gmm_fg, self.gmm_fg_bic = find_best_gmm(feature_vectors_fg)
        self.gmm_bg, self.gmm_bg_bic = find_best_gmm(feature_vectors_bg)

    def compute_node_uncertainties(self):
        for img in self.images:
            self.imgs_uncertainties_node[img] = np.zeros(len(self.imgs_segment_ids[img]))
            fg_likelihoods = self.gmm_fg.predict_proba(self.imgs_segment_feature_vectors[img])
            bg_likelihoods = self.gmm_bg.predict_proba(self.imgs_segment_feature_vectors[img])
            # Form 2-class distribution
            likelihoods = np.concatenate((fg_likelihoods, bg_likelihoods), axis=1)
            # Normalize the likelihoods
            likelihoods = likelihoods / likelihoods.sum(axis=1)[:, np.newaxis]
            # Compute the entropies of the distributions as the node uncertainties
            self.imgs_uncertainties_node[img] = [entropy(dist) for dist in likelihoods]

    def compute_edge_uncertainties(self):
        # group the foreground and background segments' feature vectors in one list
        feature_vectors_fg = [self.imgs_segment_feature_vectors[img][fg_segment] for img in self.images for fg_segment
                              in self.imgs_foreground_segments[img]]
        feature_vectors_bg = [self.imgs_segment_feature_vectors[img][bg_segment] for img in self.images for bg_segment
                              in self.imgs_background_segments[img]]

        num_fg_indices = len(feature_vectors_fg)

        feature_vectors = np.concatenate((feature_vectors_fg, feature_vectors_bg))
        assert len(feature_vectors) >= 10, "At least 10 superpixels need to be marked."

        neighbours = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(feature_vectors)

        for img in self.images:
            # Retrieve the indices of the nearest neighbours
            indices = neighbours.kneighbors(self.imgs_segment_feature_vectors[img], return_distance=False)

            # Compute the proportions of foreground and background neighbours
            proportion_fg = np.sum(indices <= num_fg_indices, axis=1) / 10
            proportion_bg = np.sum(indices > num_fg_indices, axis=1) / 10

            # Compute the uncertainties as the entropy of the foreground/background proportions
            self.imgs_uncertainties_edge[img] = [entropy([proportion_fg[id], proportion_bg[id]]) for id in self.imgs_segment_ids[img]]

    def perform_graph_cut(self, pairwise_term_scale=-np.infty, scale_parameter=1.0):
        # perform graph-cut for every image
        for img in self.images:
            # Create a graph of N nodes with an estimate of 5 edges per node
            num_nodes = len(self.imgs_segment_ids[img])
            graph = maxflow.Graph[float](num_nodes, num_nodes * 5)

            # Add the nodes
            nodes = graph.add_nodes(num_nodes)

            # If no scale is given initialize it as -infinity and set it to the largest unary term energy
            if pairwise_term_scale == -np.infty:
                compute_scale = True
            else:
                compute_scale = False

            energies_fg = np.zeros(len(self.imgs_segment_ids[img]))
            energies_bg = np.zeros(len(self.imgs_segment_ids[img]))
            edges = [dict() for i in self.imgs_segment_ids[img]]

            # Initialize match terms: energy of assigning node to foreground or background
            for i, fv in enumerate(self.imgs_segment_feature_vectors[img]):
                # set energy based on weighted log probability
                energies_fg[i] = self.gmm_fg.score_samples([fv])[0]
                energies_bg[i] = self.gmm_bg.score_samples([fv])[0]
                graph.add_tedge(nodes[i], energies_fg[i], energies_bg[i])
                # Set pairwise_term_scale to largest energy
                if compute_scale:
                    if pairwise_term_scale < abs(energies_fg[i]):
                        pairwise_term_scale = abs(energies_fg[i])
                    if pairwise_term_scale < abs(energies_bg[i]):
                        pairwise_term_scale = abs(energies_bg[i])

            # Initialize smoothness terms: energy between neighbors
            for i in self.imgs_segment_ids[img]:  # Loop over every segment
                fv = self.imgs_segment_feature_vectors[img][i]  # feature vector for segment
                for nbr in self.imgs_segment_neighbors[img][i]:  # For every neighbor of the segment
                    # Create two edges between segment and its neighbor with cost based on histogram matching
                    fv_neighbor = self.imgs_segment_feature_vectors[img][nbr]  # feature vector of segment's neighbor
                    edges[i][nbr] = pairwise_term_scale * (np.e ** (- scale_parameter * abs(euclidean(fv, fv_neighbor))))
                    edges[nbr][i] = pairwise_term_scale * (np.e ** (- scale_parameter * abs(euclidean(fv_neighbor, fv))))
                    graph.add_edge(nodes[i], nodes[nbr], edges[i][nbr], edges[nbr][i])

            graph.maxflow()

            graph_cut = graph.get_grid_segments(nodes)

            # Initialize uncertainties array
            self.imgs_uncertainties_graph_cut[img] = np.zeros(len(self.imgs_segment_ids[img]))

            # Compute uncertainties of the graph-cut as the difference in energy between the assignments
            for i in self.imgs_segment_ids[img]:
                energy_difference = abs(energies_fg[i] - energies_bg[i])
                for nbr in self.imgs_segment_neighbors[img][i]:
                    if graph_cut[i] != graph_cut[nbr]:
                        energy_difference += abs(edges[i][nbr]) + abs(edges[nbr][i])
                self.imgs_uncertainties_graph_cut[img][i] = energy_difference

            # Get a bool mask of the pixels for a given selection of superpixel IDs
            self.imgs_cosegmented[img] = np.where(np.isin(self.imgs_segmentation[img], np.nonzero(graph_cut)), True,
                                                  False)

    # To be used after superpixel segmentation and feature extraction.
    # Splits superpixels up up in num_clusters using the given method.
    def perform_clustering(self, num_clusters=2, method='kmeans'):
        data = []
        # combine the feature vectors into one list
        for img in self.images:
            for segment in self.imgs_segment_ids[img]:
                data.append(self.imgs_segment_feature_vectors[img][segment])

        # find the indices of each part in the data list
        indices = np.cumsum([len(self.imgs_segment_ids[img]) for img in self.images])

        if method is 'spectral':
            segmentations = clustering.spectral(data, num_clusters)
        elif method is 'kmeans':
            segmentations = clustering.k_means(data, num_clusters)
        else:
            print('Unknown clustering method: ' + method)
            return

        for i, img in enumerate(self.images):
            # Get segmentation for current image
            segmentation = segmentations[(indices[i - 1] % indices[-1]):indices[i]]
            # For each pixel in superpixel segmentation look up the cluster of its superpixel
            self.imgs_cosegmented[img] = [segmentation[pixel] for pixel in self.imgs_segmentation[img]]

    def get_segment_boundaries(self, img_path):
        return find_boundaries(self.imgs_segmentation[img_path])

    # Returns a binary mask after cosegmentation of the image at image_path of the segments in segments
    def get_coseg_mask(self, image_path, segments=None):
        if segments is None:
            segments = np.unique(self.imgs_cosegmented[image_path])
        return np.isin(self.imgs_cosegmented[image_path], segments)

    # write the segmented images to specified folder
    def save_segmented_images(self, folder):
        for img in self.imgs_segmentation:
            slic.save_superpixel_image(self.imgs_bgr[img], self.imgs_segmentation[img],
                                       folder + '/' + img.split('/')[-1])

    def plot_cosegmentations(self, folder_path):
        for img in self.images:
            plt.subplot(1, 2, 2), plt.xticks([]), plt.yticks([])
            plt.title('segmentation')
            plt.imshow(self.imgs_cosegmented[img])
            plt.subplot(1, 2, 1), plt.xticks([]), plt.yticks([])
            superpixels = mark_boundaries(self.imgs_bgr[img], self.imgs_segmentation[img])
            marking = cv2.imread(folder_path + 'markings/' + img.split('/')[-1])
            if marking is not None:
                superpixels[marking[:, :, 0] < 200] = (1, 0, 0)
                superpixels[marking[:, :, 2] < 200] = (0, 0, 1)
            plt.imshow(superpixels)
            plt.title("Superpixels + markings")

            plt.savefig("output/segmentation/" + img.split('/')[-1], bbox_inches='tight', dpi=96)
            plt.clf()


if __name__ == '__main__':

    folder_path = '../icoseg_images/018 Agra Taj Mahal-Inde du Nord 2004-Mhln/'
    folder_path = '../icoseg_images/043 Christ the Redeemer-Rio de Janeiro-Leonardo Paris/'
    folder_path = '../icoseg_images/025 Airshows-helicopter/'

    image_paths = [folder_path + file for file in listdir(folder_path) if path.isfile(path.join(folder_path, file))]

    alg = Algorithms(image_paths)

    # Segment the images into superpixels using slic and compute for each superpixel a list of its neighbors and its center
    alg.compute_superpixels_slic(num_segments=500, compactness=20.0, max_iter=10, sigma=0)
    alg.compute_neighbors()     # required before doing graph-cut
    alg.compute_centers()   # required before extracting sift feature

    # alg.save_segmented_images('output/superpixel')

    # Extract features
    alg.compute_feature_vectors(means_bgr=True, means_hsv=True,
                                h_hist=True, h_hist_bins=5, h_hist_entropy=True,
                                s_hist=True, s_hist_bins=3, s_hist_entropy=True,
                                hs_hist=False, hs_hist_bins_h=5, hs_hist_bins_s=3,
                                sift=False, sift_kp_size=32.0,
                                hog=False, hog_winSize=(32, 32), hog_blockSize=(16, 16), hog_blockStride=(8, 8),
                                hog_cellSize=(8, 8), hog_bins=9)

    # Retrieve foreground and background segments from marking images in markings folder
    # marking images should be white with red pixels indicating foreground and blue pixels indicating background and
    # have the same name as the image they are markings for
    for image in image_paths:
        marking = cv2.imread(folder_path + 'markings/' + image.split('/')[-1])
        if marking is not None:
            fg_segments = np.unique(alg.imgs_segmentation[image][marking[:, :, 0] < 200])
            bg_segments = np.unique(alg.imgs_segmentation[image][marking[:, :, 2] < 200])
            alg.set_fg_segments(image, fg_segments)
            alg.set_bg_segments(image, bg_segments)

    alg.compute_gmm(components_range=range(5, 9), n_init=10)
    print('Foreground GMM: ', alg.gmm_fg)
    print('BIC: ', alg.gmm_fg_bic)
    print('Background GMM:', alg.gmm_bg)
    print('BIC: ', alg.gmm_bg_bic)

    alg.compute_node_uncertainties()
    alg.compute_edge_uncertainties()

    #alg.perform_clustering(6, 'kmeans')
    alg.perform_graph_cut(pairwise_term_scale=-np.infty, scale_parameter=1.0)

    for image in image_paths:
        cv2.imwrite('output/masks/'+image.split('/')[-1], np.uint8(alg.get_coseg_mask(image, 0)*255))

    alg.plot_cosegmentations(folder_path)
'''
    # Examples on how to create histograms
    image_bgr = alg.imgs_bgr[alg.images[0]]
    mask = alg.imgs_segmentation[alg.images[0]] == 0

    hist = histograms.get_hs_histogram(image_bgr=image_bgr, mask=None, bins_h=20, bins_s=20)
    histograms.plot_hs_histogram(hist)

    hist = histograms.get_hs_histogram(image_bgr=image_bgr, mask=mask, bins_h=20, bins_s=20)
    histograms.plot_hs_histogram(hist)

    hists_bgr = [histograms.get_b_histogram(image_bgr),
                 histograms.get_g_histogram(image_bgr),
                 histograms.get_r_histogram(image_bgr)]
    histograms.plot_histograms(hists_bgr)

    print(histograms.get_bgr_histogram(image_bgr, bins_b=3, bins_g=3, bins_r=3))
'''

# TODO add initialization function for dictionaries
# TODO figure out spectral clustering?

