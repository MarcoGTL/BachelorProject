import cv2
import numpy as np
import slic
import clustering
from os import listdir, path
import maxflow
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries


class Algorithms:
    def __init__(self, image_paths):
        self.images = image_paths                                # Relative paths to the images to be used
        # Each dictionary stores the data for each image using its image path as a key
        self.imgs_float64 = dict()                               # Float64 representation of the image
        self.imgs_segmentation = dict()                          # Stores for each pixel its superpixel it belongs to
        self.imgs_segment_ids = dict()                           # List of superpixel indices
        self.imgs_segment_neighbors = dict()                     # Stores for each superpixel a list of its neighbors
        self.imgs_segment_centers = dict()
        self.imgs_segment_histograms_hsv = dict()                # Stores for each superpixel a hsv histogram
        self.imgs_segment_sift_keypoints = dict()
        self.imgs_segment_sift_descriptors = dict()
        self.imgs_segment_histograms_hsv_normalized = dict()     # Stores for each superpixel a normalized hsv histogram
        self.imgs_histograms_hsv = dict()                        # A hsv histogram of the entire image
        self.imgs_segment_feature_vectors = dict.fromkeys(image_paths, [])
        self.imgs_foreground_segments = dict.fromkeys(image_paths, [])  # List of foreground superpixels
        self.imgs_background_segments = dict.fromkeys(image_paths, [])  # List of background superpixels
        self.imgs_cosegmented = dict()            # Stores for each pixel its segment it belongs to after cosegmentation

    # generate super-pixel segments for all images using SLIC
    def compute_superpixels_slic(self, num_segments, compactness=10.0, max_iter=10, sigma=0):
        for image in self.images:
            self.imgs_float64[image] = slic.read_image_as_float64(image)
            self.imgs_segmentation[image] = slic.get_segmented_image(self.imgs_float64[image], num_segments,
                                                                     compactness, max_iter, sigma)
            self.imgs_segment_ids[image] = np.unique(self.imgs_segmentation[image])

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
    def compute_centers(self):
        for img in self.images:
            self.imgs_segment_centers[img] = []
            for i in self.imgs_segment_ids[img]:
                # Retrieve indices of segment i
                indices = np.where(self.imgs_segmentation[img] == i)
                # Center is mean of indices
                self.imgs_segment_centers[img].append((np.mean(indices[0]), np.mean(indices[1])))

    # generate hsv histograms for every segment in all images
    # also generates normalized versions
    def compute_histograms_hsv(self, bins_H=20, bins_S=20):
        for img in self.images:
            hsv = cv2.cvtColor(self.imgs_float64[img].astype('float32'), cv2.COLOR_BGR2HSV)
            self.imgs_histograms_hsv[img] = np.float32(cv2.calcHist([hsv], [0, 1], None, [bins_H, bins_S], [0, 360, 0, 1]))

            self.imgs_segment_histograms_hsv[img] = \
                np.float32([cv2.calcHist([hsv], [0, 1], np.uint8(self.imgs_segmentation[img] == i), [bins_H, bins_S],
                                         [0, 360, 0, 1]) for i in self.imgs_segment_ids[img]])

            self.imgs_segment_histograms_hsv_normalized[img] = np.float32([h / h.flatten().sum() for h in self.imgs_segment_histograms_hsv[img]])

    def compute_sift(self, keypoint_size=32.0):
        sift = cv2.xfeatures2d_SIFT.create()
        for img in self.images:
            gray = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
            self.imgs_segment_sift_keypoints[img] = [cv2.KeyPoint(self.imgs_segment_centers[img][i][1], self.imgs_segment_centers[img][i][0], keypoint_size, -1) for i in self.imgs_segment_ids[img]]
            self.imgs_segment_sift_keypoints[img], self.imgs_segment_sift_descriptors[img] = sift.compute(gray, self.imgs_segment_sift_keypoints[img])
            self.imgs_segment_sift_descriptors[img] = self.imgs_segment_sift_descriptors[img] / self.imgs_segment_sift_descriptors[img].sum()

    def compute_hog(self):
        for img in self.images:
            print('HOG not implemented')
            exit()

    # sets the foreground of the image at image_path to segments
    def set_fg_segments(self, image_path, segments):
        self.imgs_foreground_segments[image_path] = segments

    # sets the background of the image at image_path to segments
    def set_bg_segments(self, image_path, segments):
        self.imgs_background_segments[image_path] = segments

    def compute_feature_vectors(self, mode='color'):
        for img in self.images:
            self.imgs_segment_feature_vectors[img] = [[] for i in range(len(self.imgs_segment_ids[img]))]
            for segment in self.imgs_segment_ids[img]:
                # If mode is color then feature vector is the flattened color histogram
                if mode is 'color':
                    self.imgs_segment_feature_vectors[img][segment] = self.imgs_segment_histograms_hsv_normalized[img][
                        segment].flatten()
                if mode is 'sift':
                    self.imgs_segment_feature_vectors[img][segment] = self.imgs_segment_sift_descriptors[img][segment]
                if mode is 'both':
                    color_feature = self.imgs_segment_histograms_hsv_normalized[img][segment].flatten()
                    sift_feature = self.imgs_segment_sift_descriptors[img][segment]
                    self.imgs_segment_feature_vectors[img][segment] = np.concatenate((color_feature, sift_feature))

    # To be used after superpixel segmentation and feature extraction.
    # Splits the images up in num_clusters using the given method.
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

    # To be used after superpixel segmentation and feature extraction
    # Segments the image using graph cut
    def perform_graph_cut(self, hist_comp_alg=cv2.HISTCMP_KL_DIV):
        # group the foreground and background segments' feature vectors in one list
        feature_vectors_fg = [self.imgs_segment_feature_vectors[img][fg_segment] for img in self.images for fg_segment in
                       self.imgs_foreground_segments[img]]
        feature_vectors_bg = [self.imgs_segment_feature_vectors[img][bg_segment] for img in self.images for bg_segment in
                       self.imgs_background_segments[img]]

        # combine the feature vectors for into one by summing them
        fv_fg = np.sum(feature_vectors_fg, axis=0)
        fv_bg = np.sum(feature_vectors_bg, axis=0)

        # normalize the feature vector
        fv_fg = fv_fg / fv_fg.sum()
        fv_bg = fv_bg / fv_bg.sum()

        # perform graph-cut for every image
        for img in self.images:
            # Create a graph of N nodes with an estimate of 5 edges per node
            num_nodes = len(self.imgs_segment_ids[img])
            graph = maxflow.Graph[float](num_nodes, num_nodes * 5)

            # Add the nodes
            nodes = graph.add_nodes(num_nodes)

            # Initialize match terms: energy of assigning node to foreground or background
            for i, fv in enumerate(self.imgs_segment_feature_vectors[img]):
                energy_fg = 0
                energy_bg = 0
                if i in self.imgs_foreground_segments[img]:
                    energy_bg = 1000  # Node is fg -> set high energy for bg
                elif i in self.imgs_background_segments[img]:
                    energy_fg = 1000  # Node is bg -> set high energy for fg
                else:
                    # set energy based on histogram matching
                    energy_fg = cv2.compareHist(fv_fg, fv, hist_comp_alg)
                    energy_bg = cv2.compareHist(fv_bg, fv, hist_comp_alg)
                graph.add_tedge(nodes[i], energy_fg, energy_bg)

            # Initialize smoothness terms: energy between neighbors
            for i in range(len(self.imgs_segment_neighbors[img])):  # Loop over every segment
                fv = self.imgs_segment_feature_vectors[img][i]  # features for segment
                for n in self.imgs_segment_neighbors[img][i]:  # For every neighbor of the segment
                    if (n < 0) or (n > num_nodes):
                        continue
                    # Create two edges between segment and its neighbor with cost based on histogram matching
                    fv_neighbor = self.imgs_segment_feature_vectors[img][n]  # features for neighbor
                    energy_forward = 20 - cv2.compareHist(fv, fv_neighbor, hist_comp_alg)
                    energy_backward = 20 - cv2.compareHist(fv_neighbor, fv, hist_comp_alg)
                    graph.add_edge(nodes[i], nodes[n], energy_forward, energy_backward)

            graph.maxflow()

            graph_cut = graph.get_grid_segments(nodes)

            # Get a bool mask of the pixels for a given selection of superpixel IDs
            self.imgs_cosegmented[img] = np.where(np.isin(self.imgs_segmentation[img], np.nonzero(graph_cut)), True, False)

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
            slic.save_superpixel_image(self.imgs_float64[img], self.imgs_segmentation[img],
                                       folder + '/' + img.split('/')[-1])

    # Shows a plot of the histogram of the entire image at image_path or one of its segments
    def show_histogram(self, image_path, segment=None):
        if segment is None:
            plt.title(image_path.split('/')[-1])
            histogram = self.imgs_histograms_hsv[image_path]
        else:
            plt.title(image_path.split('/')[-1] + '    segment ' + str(segment))
            histogram = self.imgs_segment_histograms_hsv_normalized[image_path][segment]
        plt.imshow(histogram, interpolation='nearest')
        plt.xlabel('Saturation bins')
        plt.ylabel('Hue bins')
        plt.show()
        plt.clf()

    def plot_cosegmentations(self, folder_path):
        for img in self.images:
            plt.subplot(1, 2, 2), plt.xticks([]), plt.yticks([])
            plt.title('segmentation')
            plt.imshow(self.imgs_cosegmented[img])
            plt.subplot(1, 2, 1), plt.xticks([]), plt.yticks([])
            superpixels = mark_boundaries(self.imgs_float64[img], self.imgs_segmentation[img])
            marking = cv2.imread(folder_path + 'markings/' + img.split('/')[-1])
            if marking is not None:
                superpixels[marking[:, :, 0] < 200] = (1, 0, 0)
                superpixels[marking[:, :, 2] < 200] = (0, 0, 1)
            plt.imshow(superpixels)
            plt.title("Superpixels + markings")

            plt.savefig("output/segmentation/" + img.split('/')[-1], bbox_inches='tight', dpi=96)
            plt.clf()


if __name__ == '__main__':

    folder_path = '../images_icoseg/018 Agra Taj Mahal-Inde du Nord 2004-Mhln/'
    folder_path = '../images_icoseg/043 Christ the Redeemer-Rio de Janeiro-Leonardo Paris/'
    folder_path = '../images_icoseg/025 Airshows-helicopter/'

    image_paths = [folder_path + file for file in listdir(folder_path) if path.isfile(path.join(folder_path, file))]

    alg = Algorithms(image_paths)

    # Segment the images into superpixels using slic and compute for each superpixel a list of its neighbors
    alg.compute_superpixels_slic(num_segments=500, compactness=20.0, max_iter=10, sigma=0)
    alg.compute_neighbors()
    alg.compute_centers()

    # alg.save_segmented_images('output/superpixel')

    # Extract features
    alg.compute_sift(keypoint_size=32.0)
    alg.compute_histograms_hsv(bins_H=30, bins_S=1)

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

    alg.compute_feature_vectors(mode='both')

    # alg.perform_clustering(2, 'spectral')
    alg.perform_graph_cut()

    for image in image_paths:
        cv2.imwrite('output/masks/'+image.split('/')[-1], np.uint8(alg.get_coseg_mask(image, 0)*255))

    alg.plot_cosegmentations(folder_path)

    # alg.show_histogram('images/bear1.jpg')
    # alg.show_histogram('images/bear1.jpg', 1)

# TODO

# Graph-cut with sift
#

# Fix show histogram overwriting
# Implement Sift and HOG into cosegmentation pipeline
# Add support for different color spaces (see https://www.researchgate.net/publication/221453363_A_Comparison_Study_of_Different_Color_Spaces_in_Clustering_Based_Image_Segmentation)

