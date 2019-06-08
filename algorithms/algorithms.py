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
from scipy.stats import entropy


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

    # generate super-pixel segments for all images using SLIC
    def compute_superpixels_slic(self, num_segments, compactness=10.0, max_iter=10, sigma=0):
        for img in self.images:
            self.imgs_bgr[img] = cv2.imread(img, flags=cv2.IMREAD_COLOR)
            self.imgs_segmentation[img] = slic.get_segmented_image(self.imgs_bgr[img], num_segments,
                                                                   compactness, max_iter, sigma)
            self.imgs_segment_ids[img] = np.unique(self.imgs_segmentation[img])

    # compute the neighbor segments of each segment
    def compute_neighbors(self):
        for img in self.images:
            vs_right = np.vstack([self.imgs_segmentation[img][:, :-1].ravel(), self.imgs_segmentation[img][:, 1:].ravel()])
            vs_below = np.vstack([self.imgs_segmentation[img][:-1, :].ravel(), self.imgs_segmentation[img][1:, :].ravel()])
            neighbor_edges = np.unique(np.hstack([vs_right, vs_below]), axis=1)
            self.imgs_segment_neighbors[img] = [[] for i in self.imgs_segment_ids[img]]
            for i in range(len(neighbor_edges[0])):
                if neighbor_edges[0][i] != neighbor_edges[1][i]:
                    self.imgs_segment_neighbors[img][neighbor_edges[0][i]].append(neighbor_edges[1][i])
                    self.imgs_segment_neighbors[img][neighbor_edges[1][i]].append(neighbor_edges[0][i])

    # compute the center of each segment
    # TODO this may be outside of the superpixel for odd shapes
    def compute_centers(self):
        for img in self.images:
            self.imgs_segment_centers[img] = []
            for i in self.imgs_segment_ids[img]:
                # Retrieve indices of segment i
                indices = np.where(self.imgs_segmentation[img] == i)
                # Center is mean of indices
                self.imgs_segment_centers[img].append((np.median(indices[0]), np.median(indices[1])))

    # sets the foreground of the image at image_path to segments
    def set_fg_segments(self, image_path, segments):
        self.imgs_foreground_segments[image_path] = segments

    # sets the background of the image at image_path to segments
    def set_bg_segments(self, image_path, segments):
        self.imgs_background_segments[image_path] = segments

    def compute_feature_vectors(self, mode='color', bins_h=5, bins_s=3, kp_size=32.0):
        # TODO change order of loop and if statements
        for img in self.images:
            self.imgs_segment_feature_vectors[img] = [0 for id in self.imgs_segment_ids[img]]
            for segment in self.imgs_segment_ids[img]:
                # If mode is color then feature vector is the flattened color histogram
                if mode is 'color':
                    feature = features.get_color_feature(self.imgs_bgr[img], self.imgs_segmentation[img] == segment, bins_h, bins_s)
                elif mode is 'hsv':
                    feature = features.get_hsv_histogram_feature(self.imgs_bgr[img], self.imgs_segmentation[img] == segment, bins_h, bins_s)
                elif mode is 'sift':
                    feature = features.get_sift_feature(self.imgs_bgr[img], self.imgs_segment_centers[img][segment][1],
                                                        self.imgs_segment_centers[img][segment][0], kp_size)
                else:
                    print('Unknown feature mode: ' + mode)
                    return
                self.imgs_segment_feature_vectors[img][segment] = feature

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

    # Fits several Gaussian Mixture Model to the foreground and background superpixels depending on the range given and chooses the best fit
    def compute_gmm(self, n_init=10, num_components_range=range(5, 9)):
        # group the foreground and background segments' feature vectors in one list
        feature_vectors_fg = [self.imgs_segment_feature_vectors[img][fg_segment] for img in self.images for fg_segment
                              in self.imgs_foreground_segments[img]]
        feature_vectors_bg = [self.imgs_segment_feature_vectors[img][bg_segment] for img in self.images for bg_segment
                              in self.imgs_background_segments[img]]

        def find_best_gmm(X):
            lowest_bic = np.infty
            for n_components in num_components_range:
                # Fit a Gaussian mixture with EM
                gmm = GaussianMixture(n_components=n_components, covariance_type='full', n_init=10)
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
        # TODO
        print('Not yet implemented')

    def perform_graph_cut(self, pairwise_term_scale=-np.infty, scale_parameter=1.0):
        # perform graph-cut for every image
        for img in self.images:
            # Create a graph of N nodes with an estimate of 5 edges per node
            num_nodes = len(self.imgs_segment_ids[img])
            graph = maxflow.Graph[float](num_nodes, num_nodes * 5)

            # Add the nodes
            nodes = graph.add_nodes(num_nodes)

            # Initialize uncertainties array
            self.imgs_uncertainties_graph_cut[img] = np.zeros(len(self.imgs_segment_ids[img]))

            # If no scale is given initialize it as -infinity and set it to the largest unary term energy
            if pairwise_term_scale == -np.infty:
                compute_scale = True
            else:
                compute_scale = False

            # Initialize match terms: energy of assigning node to foreground or background
            for i, fv in enumerate(self.imgs_segment_feature_vectors[img]):
                # set energy based on weighted log probability
                energy_fg = self.gmm_fg.score_samples([fv])[0]
                energy_bg = self.gmm_bg.score_samples([fv])[0]
                graph.add_tedge(nodes[i], energy_fg, energy_bg)
                # Initialize this superpixels graph cut uncertainty as the difference in fg and bg energy
                self.imgs_uncertainties_graph_cut[img][i] = abs(energy_fg - energy_bg)
                # Set pairwise_term_scale to largest energy
                if compute_scale:
                    if pairwise_term_scale < abs(energy_fg):
                        pairwise_term_scale = abs(energy_fg)
                    if pairwise_term_scale < abs(energy_bg):
                        pairwise_term_scale = abs(energy_bg)

            # Initialize smoothness terms: energy between neighbors
            for id in self.imgs_segment_ids[img]:  # Loop over every segment
                fv = self.imgs_segment_feature_vectors[img][id]  # feature vector for segment
                for n in self.imgs_segment_neighbors[img][id]:  # For every neighbor of the segment
                    # Create two edges between segment and its neighbor with cost based on histogram matching
                    fv_neighbor = self.imgs_segment_feature_vectors[img][n]  # feature vector of segment's neighbor
                    energy_forward = pairwise_term_scale * (np.e ** (- scale_parameter * abs(cv2.compareHist(fv, fv_neighbor, cv2.HISTCMP_KL_DIV))))
                    energy_backward = pairwise_term_scale * (np.e ** (- scale_parameter * abs(cv2.compareHist(fv_neighbor, fv, cv2.HISTCMP_KL_DIV))))
                    graph.add_edge(nodes[id], nodes[n], energy_forward, energy_backward)

            graph.maxflow()

            graph_cut = graph.get_grid_segments(nodes)

            # Get a bool mask of the pixels for a given selection of superpixel IDs
            self.imgs_cosegmented[img] = np.where(np.isin(self.imgs_segmentation[img], np.nonzero(graph_cut)), True,
                                                  False)

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
    alg.compute_feature_vectors(mode='color', bins_h=5, bins_s=3, kp_size=32.0)

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

    alg.compute_gmm(n_init=10, num_components_range=range(5, 9))
    print('Foreground GMM: ', alg.gmm_fg)
    print('BIC: ', alg.gmm_fg_bic)
    print('Background GMM:', alg.gmm_bg)
    print('BIC: ', alg.gmm_bg_bic)

    alg.compute_node_uncertainties()

    # alg.perform_clustering(6, 'kmeans')
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
# TODO

# Add initialization function for dictionaries
# Uncertainty-based cues
# More features
# Smoothness term

# Fix show histogram overwriting
# Implement Sift and HOG into cosegmentation pipeline
# Add support for different color spaces (see https://www.researchgate.net/publication/221453363_A_Comparison_Study_of_Different_Color_Spaces_in_Clustering_Based_Image_Segmentation)

