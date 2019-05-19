import cv2
import numpy as np
import slic
import sift
import histograms
import argparse
import copy
import maxflow
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries


class Algorithms:
    def __init__(self, image_paths):
        self.images = image_paths
        self.imgs_float64 = dict()
        self.imgs_segmentation = dict()
        self.imgs_segment_ids = dict()
        self.imgs_segment_neighbors = dict()
        self.imgs_segment_histograms_hsv = dict()
        self.imgs_segment_histograms_hsv_normalized = dict()
        self.imgs_histograms_hsv = dict()
        self.imgs_foreground_segments = dict.fromkeys(image_paths, [])
        self.imgs_background_segments = dict.fromkeys(image_paths, [])
        self.imgs_cosegmented = dict()

    # generate super-pixel segments for all images using SLIC
    def compute_superpixels_slic(self, num_segments, compactness=10.0, max_item=10, sigma=5):
        for image in self.images:
            self.imgs_float64[image] = slic.read_image_as_float64(image)
            self.imgs_segmentation[image] = slic.get_segmented_image(self.imgs_float64[image], num_segments,
                                                                     compactness, max_item, sigma)
            self.imgs_segment_ids[image] = np.unique(self.imgs_segmentation[image])

    # generate hsv histograms for every segment in all images
    # also generates normalized versions
    def compute_histograms_hsv(self, bins_H=20, bins_S=20, range_H=None, range_S=None):
        if range_H is None:
            range_H = [0, 360]
        if range_S is None:
            range_S = [0, 1]
        for img in self.images:
            hsv = cv2.cvtColor(self.imgs_float64[img].astype('float32'), cv2.COLOR_BGR2HSV)
            self.imgs_histograms_hsv[img] = np.float32(cv2.calcHist([hsv], [0, 1], None, [bins_H, bins_S], range_H+range_S))

            plt.imshow(self.imgs_histograms_hsv[img], interpolation='nearest')
            plt.xlabel('Saturation')
            plt.ylabel('Hue')
            plt.show()

            self.imgs_segment_histograms_hsv[img] = \
                np.float32([cv2.calcHist([hsv], [0, 1], np.uint8(self.imgs_segmentation[img] == i), [bins_H, bins_S],
                                         range_H+range_S).flatten() for i in self.imgs_segment_ids[img]])

            self.imgs_segment_histograms_hsv_normalized[img] = np.float32([h / h.sum() for h in self.imgs_segment_histograms_hsv[img]])



    # compute the neighbor segments of each segment
    def compute_neighbors(self):
        for img in self.images:
            vs_right = np.vstack([self.imgs_segmentation[img][:, :-1].ravel(), self.imgs_segmentation[img][:, 1:].ravel()])
            vs_below = np.vstack([self.imgs_segmentation[img][:-1, :].ravel(), self.imgs_segmentation[img][1:, :].ravel()])
            neighbor_edges = np.unique(np.hstack([vs_right, vs_below]), axis=1)
            self.imgs_segment_neighbors[img] = [[] for x in self.imgs_segment_ids[img]]
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

    # Perform graph cut using superpixels histograms
    def do_graph_cut(self, image_path, fgbg_hists, fgbg_superpixels):
        num_nodes = self.imgs_segment_histograms_hsv_normalized[image_path].shape[0]
        # Create a graph of N nodes, and estimate of 5 edges per node
        g = maxflow.Graph[float](num_nodes, num_nodes * 5)
        # Add N nodes
        nodes = g.add_nodes(num_nodes)

        hist_comp_alg = cv2.HISTCMP_KL_DIV

        # Smoothness term: cost between neighbors
        for i in range(len(self.imgs_segment_neighbors[image_path])):
            N = self.imgs_segment_neighbors[image_path][i]  # list of neighbor superpixels
            hi = self.imgs_segment_histograms_hsv_normalized[image_path][i]  # histogram for center
            for n in N:
                if (n < 0) or (n > num_nodes):
                    continue
                # Create two edges (forwards and backwards) with capacities based on
                # histogram matching
                hn = self.imgs_segment_histograms_hsv_normalized[image_path][n]  # histogram for neighbor
                g.add_edge(nodes[i], nodes[n], 20 - cv2.compareHist(hi, hn, hist_comp_alg),
                           20 - cv2.compareHist(hn, hi, hist_comp_alg))

        # Match term: cost to FG/BG
        for i, h in enumerate(self.imgs_segment_histograms_hsv_normalized[image_path]):
            if i in fgbg_superpixels[0]:
                g.add_tedge(nodes[i], 0, 1000)  # FG - set high cost to BG
            elif i in fgbg_superpixels[1]:
                g.add_tedge(nodes[i], 1000, 0)  # BG - set high cost to FG
            else:
                g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                            cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

        g.maxflow()
        return g.get_grid_segments(nodes)

    def compute_cosegmentations(self):
        # get cumulative BG/FG histograms, being the sum of the selected superpixel IDs normalized
        h_fg = None
        h_bg = None
        for img in self.images:
            if img in self.imgs_foreground_segments:
                # TODO this is ugly
                if h_fg is None:
                    h_fg = np.sum(self.imgs_segment_histograms_hsv[img][self.imgs_foreground_segments[img]], axis=0)
                else:
                    h_fg += np.sum(self.imgs_segment_histograms_hsv[img][self.imgs_foreground_segments[img]], axis=0)
            if img in self.imgs_background_segments:
                if h_bg is None:
                    h_bg = np.sum(self.imgs_segment_histograms_hsv[img][self.imgs_background_segments[img]], axis=0)
                else:
                    h_bg += np.sum(self.imgs_segment_histograms_hsv[img][self.imgs_background_segments[img]], axis=0)
        fg_cumulative_hist = h_fg / h_fg.sum()
        bg_cumulative_hist = h_bg / h_bg.sum()

        for img in self.images:
            foreground = self.imgs_foreground_segments[img]
            background = self.imgs_background_segments[img]

            graph_cut = self.do_graph_cut(img, (fg_cumulative_hist, bg_cumulative_hist), (foreground, background))

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


def main():
    image_paths = ['images/bear1.jpg', 'images/bear2.jpg', 'images/bear3.jpg', 'images/bear4.jpg', 'images/bear5.jpg']

    alg = Algorithms(image_paths)

    # Segment the images into superpixels using slic and compute for each superpixel a list of its neighbors
    alg.compute_superpixels_slic(num_segments=500, compactness=10.0, max_item=10, sigma=5)
    alg.compute_neighbors()

    alg.save_segmented_images('output/superpixel')

    # Extract features
    alg.compute_histograms_hsv(bins_H=20, bins_S=20, range_H=[0, 360], range_S=[0, 1])

    # Retrieve foreground and background segments from marking images in markings folder
    # marking images should be white with red pixels indicating foreground and blue pixels indicating background and
    # have the same name as the image they are markings for
    for image_path in image_paths:
        marking = cv2.imread('markings/'+image_path.split('/')[-1])
        if marking is not None:
            fg_segments = np.unique(alg.imgs_segmentation[image_path][marking[:, :, 0] != 255])
            bg_segments = np.unique(alg.imgs_segmentation[image_path][marking[:, :, 2] != 255])
            alg.set_fg_segments(image_path, fg_segments)
            alg.set_bg_segments(image_path, bg_segments)

    alg.compute_cosegmentations()

    alg.plot_cosegmentations()


if __name__ == '__main__':
    main()

# TODO
# SLIC superpixel labels containing pixels could be more efficient with numpy
# Work on histograms

