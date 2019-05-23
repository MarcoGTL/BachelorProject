import cv2
import numpy as np
import slic
import clustering
from os import listdir
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
        self.imgs_segment_histograms_hsv = dict()                # Stores for each superpixel a hsv histogram
        self.imgs_sift_keypoints = dict()
        self.imgs_sift_descriptors = dict()
        self.imgs_segment_sift_descriptors = dict()
        self.imgs_segment_histograms_hsv_normalized = dict()     # Stores for each superpixel a normalized hsv histogram
        self.imgs_histograms_hsv = dict()                        # A hsv histogram of the entire image
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

    def compute_sift(self):
        sift = cv2.xfeatures2d_SIFT.create()
        for img in self.images:
            gray = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
            self.imgs_sift_keypoints[img], self.imgs_sift_descriptors[img] = sift.detectAndCompute(gray, None)

            self.imgs_segment_sift_descriptors[img] = [np.zeros(128) for i in self.imgs_segment_ids[img]]
            # TODO experimental: Give each superpixel one sift descriptor
            for i in range(len(self.imgs_sift_keypoints[img])):
                x = int(round(self.imgs_sift_keypoints[img][i].pt[0]))
                y = int(round(self.imgs_sift_keypoints[img][i].pt[1]))
                self.imgs_segment_sift_descriptors[img][self.imgs_segmentation[img][y][x]] = self.imgs_sift_descriptors[img][i]


    # Shows a plot of the histogram of the entire image at image_path or one of its segments
    def show_histogram(self, image_path, segment=None):
        if segment is None:
            plt.title(image_path.split('/')[-1])
            plt.imshow(self.imgs_histograms_hsv[image_path], interpolation='nearest')
        else:
            plt.title(image_path.split('/')[-1] + '    segment ' + str(segment))
            plt.imshow(self.imgs_segment_histograms_hsv[image_path][segment], interpolation='nearest')
        plt.xlabel('Saturation bins')
        plt.ylabel('Hue bins')
        plt.show()
        plt.clf()

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

    # sets the foreground of the image at image_path to segments
    def set_fg_segments(self, image_path, segments):
        self.imgs_foreground_segments[image_path] = segments

    # sets the background of the image at image_path to segments
    def set_bg_segments(self, image_path, segments):
        self.imgs_background_segments[image_path] = segments

    # To be used after superpixel segmentation and feature extraction
    # Segments the image using graph cut
    def perform_graph_cut(self):
        def compute_cumulative_histograms():
            # for each image sum up the histograms of the chosen segments
            histograms_fg = [
                np.sum([h.flatten() for h in self.imgs_segment_histograms_hsv[img][self.imgs_foreground_segments[img]]],
                       axis=0) for img in self.images]
            histograms_bg = [
                np.sum([h.flatten() for h in self.imgs_segment_histograms_hsv[img][self.imgs_background_segments[img]]],
                       axis=0) for img in self.images]

            # combine the histograms for each image into one
            histogram_fg = np.sum(histograms_fg, axis=0)
            histogram_bg = np.sum(histograms_bg, axis=0)

            # normalize the histograms to get the final cumulative histograms
            return histogram_fg / histogram_fg.sum(), histogram_bg / histogram_bg.sum()

        # get cumulative BG/FG histograms, being the sum of the selected superpixel IDs normalized
        fg_hist, bg_hist = compute_cumulative_histograms()

        def do_graph_cut(image_path, fg_hist, bg_hist, hist_comp_alg=cv2.HISTCMP_KL_DIV):
            num_nodes = len(self.imgs_segment_ids[image_path])

            # Create a graph of N nodes with an estimate of 5 edges per node
            g = maxflow.Graph[float](num_nodes, num_nodes * 5)

            # Add N nodes
            nodes = g.add_nodes(num_nodes)

            # Initialize smoothness terms: energy between neighbors
            for i in range(len(self.imgs_segment_neighbors[image_path])):
                N = self.imgs_segment_neighbors[image_path][i]  # list of neighbor superpixels
                hi = self.imgs_segment_histograms_hsv_normalized[image_path][i].flatten()  # histogram for center
                for n in N:
                    if (n < 0) or (n > num_nodes):
                        continue
                    # Create two edges (forwards and backwards) with capacities based on
                    # histogram matching
                    hn = self.imgs_segment_histograms_hsv_normalized[image_path][n].flatten()  # histogram for neighbor
                    g.add_edge(nodes[i], nodes[n], 20 - cv2.compareHist(hi, hn, hist_comp_alg),
                               20 - cv2.compareHist(hn, hi, hist_comp_alg))

            # Initialize match terms: energy of assigning node to foreground or background
            for i, h in enumerate(self.imgs_segment_histograms_hsv_normalized[image_path]):
                h = h.flatten()
                energy_fg = 0
                energy_bg = 0
                if i in self.imgs_foreground_segments[image_path]:
                    energy_bg = 1000  # Node is fg -> set high energy for bg
                elif i in self.imgs_background_segments[image_path]:
                    energy_fg = 1000  # Node is bg -> set high energy for fg
                else:
                    energy_fg = cv2.compareHist(fg_hist, h, hist_comp_alg)
                    energy_bg = cv2.compareHist(bg_hist, h, hist_comp_alg)
                g.add_tedge(nodes[i], energy_fg, energy_bg)

            g.maxflow()
            return g.get_grid_segments(nodes)

        for img in self.images:
            graph_cut = do_graph_cut(img, fg_hist, bg_hist)

            # Get a bool mask of the pixels for a given selection of superpixel IDs
            segmentation = np.where(np.isin(self.imgs_segmentation[img], np.nonzero(graph_cut)), True, False)

            self.imgs_cosegmented[img] = np.uint8(segmentation * 255)

    def get_segment_boundaries(self, img_path):
        return find_boundaries(self.imgs_segmentation[img_path])

    # write the segmented images to specified folder
    def save_segmented_images(self, folder):
        for image in self.imgs_segmentation:
            slic.save_superpixel_image(self.imgs_float64[image], self.imgs_segmentation[image],
                                       folder + '/' + image.split('/')[-1])

    def plot_cosegmentations(self):
        for img in self.images:
            plt.subplot(1, 2, 2), plt.xticks([]), plt.yticks([])
            plt.title('segmentation')
            plt.imshow(self.imgs_cosegmented[img])
            plt.subplot(1, 2, 1), plt.xticks([]), plt.yticks([])
            superpixels = mark_boundaries(self.imgs_float64[img], self.imgs_segmentation[img])
            marking = cv2.imread('markings/' + img.split('/')[-1])
            if marking is not None:
                superpixels[marking[:, :, 0] != 255] = (1, 0, 0)
                superpixels[marking[:, :, 2] != 255] = (0, 0, 1)
            plt.imshow(superpixels)
            plt.title("Superpixels + markings")

            plt.savefig("output/segmentation/" + img.split('/')[-1], bbox_inches='tight', dpi=96)
            plt.clf()

    # To be used after superpixel segmentation and feature extraction.
    # Splits the images up in num_clusters using the given method.
    def perform_clustering(self, num_clusters=2, method='spectral', type='histogram'):
        data = []
        for img in self.images:
            if type is 'histogram':
                for h in self.imgs_segment_histograms_hsv_normalized[img]:
                    data.append(h.flatten())
            elif type is 'sift':
                for s in self.imgs_segment_sift_descriptors[img]:
                    data.append(s)
            elif type is 'both':
                for i in range(len(self.imgs_segment_sift_descriptors[img])):
                    h = self.imgs_segment_histograms_hsv_normalized[img][i].flatten()
                    s = self.imgs_segment_sift_descriptors[img][i]
                    data.append(np.concatenate((h, s)))

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
            segmentation = segmentations[(indices[i-1] % indices[-1]):indices[i]]
            # For each pixel in superpixel segmentation look up the cluster of its superpixel
            self.imgs_cosegmented[img] = [segmentation[pixel] for pixel in alg.imgs_segmentation[img]]

    # Returns a binary mask after cosegmentation of the image at image_path of the segments in segments
    def get_coseg_mask(self, image_path, segments=None):
        if segments is None:
            segments = np.unique(self.imgs_cosegmented[image_path])
        return np.isin(self.imgs_cosegmented[image_path], segments)


if __name__ == '__main__':

    folder_path = '../images_icoseg/043 Christ the Redeemer-Rio de Janeiro-Leonardo Paris/'
    folder_path = '../images_icoseg/018 Agra Taj Mahal-Inde du Nord 2004-Mhln/'
    image_paths = [folder_path + name for name in listdir(folder_path)]

    alg = Algorithms(image_paths)

    # Segment the images into superpixels using slic and compute for each superpixel a list of its neighbors
    alg.compute_superpixels_slic(num_segments=500, compactness=20.0, max_iter=10, sigma=0)
    alg.compute_neighbors()

    # alg.save_segmented_images('output/superpixel')

    # Extract features
    alg.compute_sift()

    alg.compute_histograms_hsv(bins_H=20, bins_S=20)

    # Retrieve foreground and background segments from marking images in markings folder
    # marking images should be white with red pixels indicating foreground and blue pixels indicating background and
    # have the same name as the image they are markings for
    for image in image_paths:
        marking = cv2.imread('markings/'+image.split('/')[-1])
        if marking is not None:
            fg_segments = np.unique(alg.imgs_segmentation[image][marking[:, :, 0] != 255])
            bg_segments = np.unique(alg.imgs_segmentation[image][marking[:, :, 2] != 255])
            alg.set_fg_segments(image, fg_segments)
            alg.set_bg_segments(image, bg_segments)

    alg.perform_clustering(2, 'kmeans', 'both')

    for image in image_paths:
        cv2.imwrite('output/masks/'+image.split('/')[-1], np.uint8(alg.get_coseg_mask(image, 0)*255))

    # alg.compute_cosegmentations_graph_cut()

    alg.plot_cosegmentations()

    # alg.show_histogram('images/bear1.jpg')
    # alg.show_histogram('images/bear1.jpg', 1)

# TODO

# Fix show histogram overwriting
# Implement Sift and HOG into cosegmentation pipeline
# Allow for unsupervised segmentation
# Add support for different color spaces (see https://www.researchgate.net/publication/221453363_A_Comparison_Study_of_Different_Color_Spaces_in_Clustering_Based_Image_Segmentation)


# TODO nice to have
# SLIC superpixel labels containing pixels could be more efficient with numpy

