import cv2
import numpy as np
from featureextraction import FeatureExtraction
import histograms
from os import listdir, path
import maxflow
from skimage.segmentation import slic, mark_boundaries, find_boundaries
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt


class Pipeline:
    """
    A class containing a collection of all the functions that are used in the cosegmentation pipeline.

    The attributes of this class consist mostly of dictionaries with the image paths as keys. The outcomes of each step
    are stored in here.

    At the end of this file is a main function that demonstrates how this pipeline can be used.

    Required Packages:
        opencv-contrib-python (Using 3.4.2.17, newer versions do not include SIFT)
        numpy
        PyMaxflow
        scikit-image
        scikit-learn
        scipy
        matplotlib

    Args:
        image_paths ([str]): A list of image paths of the images to be used in the pipeline

    Attributes:
        images ([str]): Relative paths to the images to be used
        images_feature_extraction (dict of str: FeatureExtraction()): A dictionary containing for each image
            a FeatureExtraction object.
        images_segmented (dict of str: ndarray): 2D arrays of numpy.int64 values indicating for each pixel the
            superpixels they belong to after slic.
        images_superpixels (dict of str: ndarray): 1D arrays of numpy.int64 values listing the superpixel indices
            that are present in each image.
        images_superpixels_neighbours (dict of str: [set()]): Contains for each image a list of sets of numpy.int64
            values representing the neighbours of each superpixel
        images_superpixels_center (dict of str: [tuple()]): Contains for each image a list of (x, y) tuples of
            numpy.int64 values representing the x and y coordinates of the superpixel centers.

        images_superpixels_feature_vector (dict of str: [ndarray]): A list containing for each superpixel
        a feature vector consisting of a 1D array of numpy.float32 values.

        images_superpixels_foreground (dict of str: [int]): List of superpixel labels set as foreground
        images_superpixels_background (dict of str: [int]): List of superpixel labels set as background

        gmm_foreground (GaussianMixture): A Gaussian Mixture Model for the foreground
        gmm_background (GaussianMixture): A Gaussian Mixture Model for the background

        gmm_foreground_bic (float64): Bayesian information criterion for the foreground GMM model
        gmm_background_bic (float64): Bayesian information criterion for the background GMM model

        images_superpixels_uncertainties_node (dict of str: [float]): List of node uncertainty scores of each superpixel
        images_superpixels_uncertainties_edge (dict of str: [float]): List of edge uncertainty scores of each superpixel
        images_superpixels_uncertainties_graph_cut (dict of str: [float]): List of uncertainty scores of each superpixel
            based on difference in energy between given label and its opposite.

        images_cosegmented (dict of str: ndarray): 2D array of numpy.int32 values indicating for every pixel its label
            after cosegmentation.

    """
    def __init__(self, image_paths: [str]):
        self.images = image_paths
        self.images_feature_extraction = {img: FeatureExtraction(img) for img in self.images}

        self.images_segmented = dict.fromkeys(self.images)
        self.images_superpixels = dict.fromkeys(self.images)
        self.images_superpixels_neighbours = dict.fromkeys(self.images)
        self.images_superpixels_center = dict.fromkeys(self.images, [])

        self.images_superpixels_feature_vector = dict()

        self.images_superpixels_foreground = dict.fromkeys(image_paths, [])
        self.images_superpixels_background = dict.fromkeys(image_paths, [])

        self.gmm_foreground = None
        self.gmm_background = None
        self.gmm_foreground_bic = 0
        self.gmm_background_bic = 0

        self.images_superpixels_uncertainties_node = dict()
        self.images_superpixels_uncertainties_edge = dict()
        self.images_superpixels_uncertainties_graph_cut = dict()

        self.images_cosegmented = dict()

    def compute_superpixels(self, num_segments, compactness=10.0, max_iter=10, sigma=0):
        """
        Generate superpixel segmentations using SLIC. The result is stored in images_segmented and images_superpixels

        Args:
            num_segments (int): The approximate amount of superpixels created in each image
            compactness (float): Balances color proximity and space proximity. Higher results in more square superpixels
            max_iter (int): Maximum number of iterations of k-means
            sigma (float): Width of Gaussian smoothing kernel for optional pre-processing (0 for no smoothing)
        """
        for img in self.images:
            self.images_segmented[img] = slic(self.images_feature_extraction[img].image_bgr,
                                              num_segments, compactness, max_iter, sigma)
            self.images_superpixels[img] = np.unique(self.images_segmented[img])

    def compute_neighbors(self):
        """
        Computes for each superpixel a set of its neighbours and stores it in images_superpixels_neighbours.

        Note: Requires compute_superpixels
        """
        for img in self.images:
            self.images_superpixels_neighbours[img] = [set() for sp in self.images_superpixels[img]]
            for row in range(len(self.images_segmented[img]) - 1):
                for column in range(len(self.images_segmented[img][0]) - 1):
                    current = self.images_segmented[img][row][column]  # superpixel label of current pixel
                    right = self.images_segmented[img][row][column + 1]  # superpixel label of pixel right of current
                    below = self.images_segmented[img][row + 1][column]  # superpixel label of pixel below current
                    if current != right:
                        self.images_superpixels_neighbours[img][current].add(right)
                        self.images_superpixels_neighbours[img][right].add(current)
                    if current != below:
                        self.images_superpixels_neighbours[img][current].add(below)
                        self.images_superpixels_neighbours[img][below].add(current)

    # compute the center of each superpixel
    # this method may yield locations outside superpixels with odd shapes
    def compute_centers(self):
        """
        Computes for each superpixel its center coordinates as (x, y) tuple and stores them in images_superpixels_center

        Note: Requires compute_superpixels
        """
        for img in self.images:
            for i in self.images_superpixels[img]:
                # Retrieve all indices where superpixel label equals i
                indices = np.where(self.images_segmented[img] == i)
                # Approximate the center by the medians of the indices in x and y dimension
                self.images_superpixels_center[img].append((np.median(indices[1]), np.median(indices[0])))

    def set_fg_segments(self, image_path: str, superpixel_labels: [int]):
        """
        Set an entry in images_superpixels_foreground with key image_path to superpixel_labels

        Parameters:
            image_path: The relative path to the image for which the superpixels are set as foreground
            superpixel_labels: The list of superpixels that are set to foreground
        """
        self.images_superpixels_foreground[image_path] = superpixel_labels

    # sets the background of the image at image_path to segments
    def set_bg_segments(self, image_path, superpixel_labels):
        """
        Does the same as images_superpixels_foreground but for images_superpixels_background
        """
        self.images_superpixels_background[image_path] = superpixel_labels

    def compute_feature_vectors(self, means_bgr=True, means_hsv=True,
                                h_hist=True, h_hist_bins=5, h_hist_entropy=True,
                                s_hist=True, s_hist_bins=3, s_hist_entropy=True,
                                hs_hist=False, hs_hist_bins_h=5, hs_hist_bins_s=3,
                                sift=False, sift_kp_size=32.0,
                                hog=False, hog_winSize=(32, 32), hog_blockSize=(16, 16), hog_blockStride=(8, 8),
                                hog_cellSize=(8, 8), hog_bins=9):
        """
        Compute feature vectors for each superpixel consisting of the elements that are set to True.

        The default feature vector is of length 16 consists of means of BGR, HSV,
        and a histogram with entropy of Hue (5 bins) and Saturation (3bins)

        Note: Requires compute_superpixels and compute_superpixels_center

        Parameters:
            means_bgr (bool): Include the means of BGR
            means_hsv (bool): Include the means of HSV
            h_hist (bool): Include a histogram of Hue
            h_hist_bins (int): Number of bins for Hue histogram
            h_hist_entropy (bool): Include entropy of Hue histogram
            s_hist (bool): Include a histogram of Saturation
            s_hist_bins (int): Number of bins for Saturation histogram
            s_hist_entropy (bool): Include entropy of Saturation histogram
            hs_hist (bool): Include a flattened 2D histogram of Hue and Saturation
            hs_hist_bins_h (int): Number of Hue bins for Hue/Saturation histogram
            hs_hist_bins_s (int): Number of Saturation bins for Hue/Saturation histogram
            sift (bool): Include a SIFT descriptor
            sift_kp_size (float): Diameter of the keypoint neighborhood
            hog (bool): Include a HOG descriptor
            hog_winSize ((int, int)): Size in pixels of the detection window in (width, height) of the HOG descriptor
            hog_blockSize ((int, int)): Block size of the HOG descriptor in pixels. Align to cell size
            hog_blockStride ((int, int)): Block stride of the HOG descriptor. It must be a multiple of cell size
            hog_cellSize ((int, int)): Cell size of the HOG descriptor
            hog_bins (int): Number of bins per cell of the HOG descriptor
        """
        for img in self.images:
            self.images_superpixels_feature_vector[img] = [[] for sp in self.images_superpixels[img]]
            for superpixel in self.images_superpixels[img]:
                feature_vector = []
                mask = np.uint8(self.images_segmented[img] == superpixel)
                coord_xy = (self.images_superpixels_center[img][superpixel][0],
                            self.images_superpixels_center[img][superpixel][1])
                if means_bgr:
                    feature_vector += self.images_feature_extraction[img].means_bgr(mask)
                if means_hsv:
                    feature_vector += self.images_feature_extraction[img].means_hsv(mask)
                if h_hist:
                    feature_vector += self.images_feature_extraction[img].h_hist(mask, h_hist_bins, h_hist_entropy)
                if s_hist:
                    feature_vector += self.images_feature_extraction[img].s_hist(mask, s_hist_bins, s_hist_entropy)
                if hs_hist:
                    feature_vector += self.images_feature_extraction[img].hs_hist(mask, hs_hist_bins_h, hs_hist_bins_s)
                if sift:
                    feature_vector += self.images_feature_extraction[img].sift(coord_xy[0], coord_xy[1], sift_kp_size)
                if hog:
                    feature_vector += self.images_feature_extraction[img].hog(coord_xy[0], coord_xy[1], hog_winSize,
                                                            hog_blockSize, hog_blockStride, hog_cellSize, hog_bins)
                assert len(feature_vector) > 0, 'Feature vector needs to have at least one feature'
                self.images_superpixels_feature_vector[img][superpixel] = np.asarray(feature_vector, dtype='float32')

    def compute_gmm(self, components_range=range(5, 9), n_init=10):
        """
        Fits a Gaussian Mixture Model to the feature vectors of the foreground and background superpixels and chooses
        the best fitting model. Stores the models in gmm_foreground/gmm_background and their bic scores in
        gmm_foreground_bic/gmm_background_bic.

        Note: Requires compute_feature_vectors and at least one superpixel in foreground and background markings

        Parameters:
            components_range (range(int, int) or int): Try to fit a model for these amounts of components
            n_init (int): Number of restarts per number of components
        """
        # group the foreground and background superpixels' feature vectors in one list
        feature_vectors_fg = [self.images_superpixels_feature_vector[img][fg_segment] for img in self.images for fg_segment
                              in self.images_superpixels_foreground[img]]
        feature_vectors_bg = [self.images_superpixels_feature_vector[img][bg_segment] for img in self.images for bg_segment
                              in self.images_superpixels_background[img]]

        assert len(feature_vectors_fg) > 0, 'At least one superpixel needs to be marked as foreground'
        assert len(feature_vectors_bg) > 0, 'At least one superpixel needs to be marked as background'

        def find_best_gmm(data):
            lowest_bic = np.infty
            for n_components in components_range:
                # Fit a Gaussian mixture with EM
                gmm = GaussianMixture(n_components=n_components, covariance_type='full', n_init=n_init)
                gmm.fit(data)
                bic = gmm.bic(np.asarray(data, dtype=np.float32))
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
            return best_gmm, lowest_bic

        self.gmm_foreground, self.gmm_foreground_bic = find_best_gmm(feature_vectors_fg)
        self.gmm_background, self.gmm_background_bic = find_best_gmm(feature_vectors_bg)

    def compute_node_uncertainties(self):
        """
        Compute an uncertainty score for every superpixel based on entropy of node beliefs.
        The result is stored in images_superpixels_uncertainties_node

        Note: Requires compute_gmm.
        """
        for img in self.images:
            self.images_superpixels_uncertainties_node[img] = np.zeros(len(self.images_superpixels[img]))
            fg_likelihoods = self.gmm_foreground.predict_proba(self.images_superpixels_feature_vector[img])
            bg_likelihoods = self.gmm_background.predict_proba(self.images_superpixels_feature_vector[img])
            # Form 2-class distribution
            likelihoods = np.concatenate((fg_likelihoods, bg_likelihoods), axis=1)
            # Normalize the likelihoods
            likelihoods = likelihoods / likelihoods.sum(axis=1)[:, np.newaxis]
            # Compute the entropies of the distributions as the node uncertainties
            self.images_superpixels_uncertainties_node[img] = [entropy(dist) for dist in likelihoods]

    def compute_edge_uncertainties(self):
        """
        Compute an uncertainty score for every superpixel based on the entropy of the proportion of its K (=10) nearest
        neighbours of marked superpixels. The result is stored in images_superpixels_uncertainties_edge

        Note: Requires compute_gmm.
        """
        # group the foreground and background segments' feature vectors in one list
        feature_vectors_fg = [self.images_superpixels_feature_vector[img][fg_segment] for img in self.images for fg_segment
                              in self.images_superpixels_foreground[img]]
        feature_vectors_bg = [self.images_superpixels_feature_vector[img][bg_segment] for img in self.images for bg_segment
                              in self.images_superpixels_background[img]]

        num_fg_indices = len(feature_vectors_fg)

        feature_vectors = np.concatenate((feature_vectors_fg, feature_vectors_bg))
        assert len(feature_vectors) >= 10, "At least 10 superpixels need to be marked."

        neighbours = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(feature_vectors)

        for img in self.images:
            # Retrieve the indices of the nearest neighbours
            indices = neighbours.kneighbors(self.images_superpixels_feature_vector[img], return_distance=False)

            # Compute the proportions of foreground and background neighbours
            proportion_fg = np.sum(indices <= num_fg_indices, axis=1) / 10
            proportion_bg = np.sum(indices > num_fg_indices, axis=1) / 10

            # Compute the uncertainties as the entropy of the foreground/background proportions
            self.images_superpixels_uncertainties_edge[img] = [entropy([proportion_fg[sp], proportion_bg[sp]])
                                                               for sp in self.images_superpixels[img]]

    def perform_graph_cut(self, pairwise_term_scale=-np.infty, scale_parameter=1.0):
        """
        Segments every image using graph-cut. The graph built has nodes with energies based on GMM matching, and edges
        based on euclidean distance between neighbouring superpixels' feature vectors. The resulting cosegmentation is
        stored in images_cosegmented. This function also calculates for each superpixel an uncertainty score based on
        the difference in energy between the optimal assignment and opposite and stores
        these scores in images_superpixels_uncertainties_graph_cut.

        Note: Requires compute_gmm

        parameters:
            pairwise_term_scale (float): Used to scale the pairwise term in relation to the unary term.
            scale_parameter (float): Used to adjust the strength of the response in the pairwise term value
                depending on distance.
        """
        # perform graph-cut for every image
        for img in self.images:
            # Create a graph of N nodes with an estimate of 5 edges per node
            num_nodes = len(self.images_superpixels[img])
            graph = maxflow.Graph[float](num_nodes, num_nodes * 5)

            # Add the nodes
            nodes = graph.add_nodes(num_nodes)

            # If no scale is given initialize it as -infinity and set it to the largest unary term energy
            if pairwise_term_scale == -np.infty:
                compute_scale = True
            else:
                compute_scale = False

            energies_fg = np.zeros(len(self.images_superpixels[img]))
            energies_bg = np.zeros(len(self.images_superpixels[img]))
            edges = [dict() for i in self.images_superpixels[img]]

            # Initialize match terms: energy of assigning node to foreground or background
            for sp, fv in enumerate(self.images_superpixels_feature_vector[img]):
                # set energy based on weighted log probability
                energies_fg[sp] = self.gmm_foreground.score_samples([fv])[0]
                energies_bg[sp] = self.gmm_background.score_samples([fv])[0]
                graph.add_tedge(nodes[sp], energies_fg[sp], energies_bg[sp])
                # Set pairwise_term_scale to largest energy
                if compute_scale:
                    if pairwise_term_scale < abs(energies_fg[sp]):
                        pairwise_term_scale = abs(energies_fg[sp])
                    if pairwise_term_scale < abs(energies_bg[sp]):
                        pairwise_term_scale = abs(energies_bg[sp])

            # Initialize smoothness terms: energy between neighbors
            for sp in self.images_superpixels[img]:
                fv = self.images_superpixels_feature_vector[img][sp]
                for nbr in self.images_superpixels_neighbours[img][sp]:
                    # Create two edges between superpixel and its neighbor with cost based on
                    # euclidean distance between their feature vectors
                    fv_neighbor = self.images_superpixels_feature_vector[img][nbr]
                    edges[sp][nbr] = pairwise_term_scale * (np.e ** (- scale_parameter * abs(euclidean(fv, fv_neighbor))))
                    edges[nbr][sp] = pairwise_term_scale * (np.e ** (- scale_parameter * abs(euclidean(fv_neighbor, fv))))
                    graph.add_edge(nodes[sp], nodes[nbr], edges[sp][nbr], edges[nbr][sp])

            graph.maxflow()

            graph_cut = graph.get_grid_segments(nodes)

            # Initialize uncertainties array
            self.images_superpixels_uncertainties_graph_cut[img] = [0 for sp in self.images_superpixels[img]]

            # Compute uncertainties of the graph-cut as the difference in energy between the assignments
            for sp in self.images_superpixels[img]:
                energy_difference = abs(energies_fg[sp] - energies_bg[sp])
                for nbr in self.images_superpixels_neighbours[img][sp]:
                    if graph_cut[sp] != graph_cut[nbr]:
                        energy_difference += abs(edges[sp][nbr]) + abs(edges[nbr][sp])
                self.images_superpixels_uncertainties_graph_cut[img][sp] = energy_difference

            # Get a bool mask of the pixels for a given selection of superpixel IDs
            self.images_cosegmented[img] = np.where(np.isin(self.images_segmented[img], np.nonzero(graph_cut)), 0, 1)

    def perform_k_means_clustering(self, num_clusters=2):
        """
        An unsupervised alternative to graph-cut. This function simply performs K-means clustering with all superpixel
        feature vectors. The resulting segmentation is stored in images_cosegmented.

        Note: Requires compute_feature_vectors

        Parameters:
            num_clusters: The desired number of clusters.
        """

        # Combine all feature vectors into one list
        feature_vectors = [self.images_superpixels_feature_vector[img][sp]
                           for img in self.images for sp in self.images_superpixels[img]]

        segmentations = KMeans(n_clusters=num_clusters).fit(feature_vectors).labels_

        # Find the indices of each individual part for the combined list
        indices = np.cumsum([len(self.images_superpixels[img]) for img in self.images])

        for i, img in enumerate(self.images):
            # Retrieve segmentation for a single image
            segmentation = segmentations[(indices[i - 1] % indices[-1]):indices[i]]
            # For each pixel in the image look up the segmentation label of its superpixel
            self.images_cosegmented[img] = [segmentation[pixel] for pixel in self.images_segmented[img]]

    def get_superpixel_borders_mask(self, img_path):
        """ Retrieve an image mask for the superpixel borders

        Note: Requires compute_superpixels

        Parameters:
            img_path (str): Relative path to the image

        Returns:
            An image mask of the boundaries between superpixels
        """
        return find_boundaries(self.images_segmented[img_path])

    def get_coseg_mask(self, image_path, labels=None):
        """ Retrieve an image mask of a cosegmented image excluding all segments not specified by the labels parameter.

        Parameters:
            image_path (str): Relative path to the image
            labels (int or [int]): Segment labels to include in the mask

        Returns:
             An image mask of the cosegmentation
        """
        if labels is None:
            labels = np.unique(self.images_cosegmented[image_path])
        return np.isin(self.images_cosegmented[image_path], labels)

    def plot_cosegmentations(self, folder_path):
        """ Plots for each image its superpixel segmentation with markings and a cosegmentation and saves it to 'output/segmentation/<image>'

        Parameters:
            folder_path (str): Relative path to the folder of the input images
        """
        for img in self.images:
            plt.subplot(1, 2, 2), plt.xticks([]), plt.yticks([])
            plt.title('segmentation')
            plt.imshow(self.images_cosegmented[img])
            plt.subplot(1, 2, 1), plt.xticks([]), plt.yticks([])
            superpixels = mark_boundaries(self.images_feature_extraction[img].image_bgr, self.images_segmented[img])
            marking = cv2.imread(folder_path + 'markings/' + img.split('/')[-1])
            if marking is not None:
                superpixels[marking[:, :, 0] < 200] = (1, 0, 0)
                superpixels[marking[:, :, 2] < 200] = (0, 0, 1)
            plt.imshow(superpixels)
            plt.title("Superpixels + markings")

            plt.savefig("output/segmentation/" + img.split('/')[-1], bbox_inches='tight', dpi=96)
            plt.clf()


if __name__ == '__main__':

    # folder_path = '../icoseg_images/018 Agra Taj Mahal-Inde du Nord 2004-Mhln/'
    # folder_path = '../icoseg_images/043 Christ the Redeemer-Rio de Janeiro-Leonardo Paris/'
    folder_path = '../icoseg_images/025 Airshows-helicopter/'

    # Make a list of file paths of the images inside the folder
    images = [folder_path + file for file in listdir(folder_path) if path.isfile(path.join(folder_path, file))]

    # Initialize a cosegmentation pipeline for the images
    pipeline = Pipeline(images)

    # Compute superpixel segmentation using slic and compute for each superpixel a list of its neighbors and its center
    pipeline.compute_superpixels(num_segments=500, compactness=20.0, max_iter=10, sigma=0)
    pipeline.compute_neighbors()
    pipeline.compute_centers()

    # Extract features
    pipeline.compute_feature_vectors(means_bgr=True, means_hsv=True,
                                     h_hist=True, h_hist_bins=5, h_hist_entropy=True,
                                     s_hist=True, s_hist_bins=3, s_hist_entropy=True,
                                     hs_hist=False, hs_hist_bins_h=5, hs_hist_bins_s=3,
                                     sift=False, sift_kp_size=32.0,
                                     hog=False, hog_winSize=(32, 32), hog_blockSize=(16, 16), hog_blockStride=(8, 8),
                                     hog_cellSize=(8, 8), hog_bins=9)

    # Retrieve foreground and background segments from marking images in markings folder
    # marking images should be white with red pixels indicating foreground and blue pixels indicating background and
    # have the same name as the image they are markings for
    for image in images:
        marking = cv2.imread(folder_path + 'markings/' + image.split('/')[-1])
        if marking is not None:
            fg_segments = np.unique(pipeline.images_segmented[image][marking[:, :, 0] < 200])
            bg_segments = np.unique(pipeline.images_segmented[image][marking[:, :, 2] < 200])
            pipeline.set_fg_segments(image, fg_segments)
            pipeline.set_bg_segments(image, bg_segments)

    # Fit a GMM for the foreground and background marked superpixels
    pipeline.compute_gmm(components_range=range(5, 9), n_init=10)
    print('Foreground GMM: ', pipeline.gmm_foreground)
    print('BIC: ', pipeline.gmm_foreground_bic)
    print('Background GMM:', pipeline.gmm_background)
    print('BIC: ', pipeline.gmm_background_bic)

    # Compute uncertainty scores
    pipeline.compute_node_uncertainties()
    pipeline.compute_edge_uncertainties()

    # Perform cosegmentation: either unsupervised k_means clustering or graph-cut can be used
    # pipeline.perform_k_means_clustering(num_clusters=6)
    pipeline.perform_graph_cut(pairwise_term_scale=-np.infty, scale_parameter=1.0)

    # Output the results in output folder
    pipeline.plot_cosegmentations(folder_path)
    for image in images:
        cv2.imwrite('output/masks/' + image.split('/')[-1], np.uint8(pipeline.get_coseg_mask(image, 1) * 255))

    # Examples on how to create histograms
    """
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
    """

# TODO: give different names to the graph-cut scales
