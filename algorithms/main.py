import cv2
import numpy as np
import slic
import sift
import histogram
import argparse
import copy
import coseg
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


class Algorithms:
    def __init__(self, image_paths):
        self.images = image_paths
        self.imgs_float64 = dict()
        self.imgs_segmented = dict()
        self.imgs_segment_ids = dict()
        self.imgs_segment_neighbors = dict()
        self.imgs_color_histograms = dict()
        self.imgs_color_histograms_normalized = dict()
        self.imgs_cosegmented = dict()

    # generate super-pixel segments for images in img_paths using SLIC
    def slic(self, num_segments, compactness=10.0, max_item=10, sigma=5):
        for image in self.images:
            self.imgs_float64[image] = slic.read_image_as_float64(image)
            self.imgs_segmented[image] = slic.get_segmented_image(self.imgs_float64[image], num_segments,
                                                                  compactness, max_item, sigma)
            self.imgs_segment_ids[image] = np.unique(self.imgs_segmented[image])

    def perform_cosegmentation(self, fg_segments, bg_segments):
        for img in self.images:
            centers = np.array([np.mean(np.nonzero(self.imgs_segmented[img] == i), axis=1)
                                for i in self.imgs_segment_ids[img]])

            self.imgs_segment_neighbors[img] = Delaunay(centers).vertex_neighbor_vertices

            hsv = cv2.cvtColor(self.imgs_float64[img].astype('float32'), cv2.COLOR_BGR2HSV)
            bins = [20, 20]  # H = S = 20
            ranges = [0, 360, 0, 1]  # H: [0, 360], S: [0, 1]
            self.imgs_color_histograms[img] = np.float32(
                [cv2.calcHist([hsv], [0, 1], np.uint8(self.imgs_segmented[img] == i), bins, ranges).flatten()
                 for i in self.imgs_segment_ids[img]])
            self.imgs_color_histograms_normalized[img] = coseg.normalize_histograms(
                self.imgs_color_histograms[img])

        # get cumulative BG/FG histograms, being the sum of the selected superpixel IDs and then normalized
        h_fg = None
        h_bg = None
        for img in self.images:
            if img in fg_segments:
                if h_fg is None:
                    h_fg = np.sum(self.imgs_color_histograms[img][fg_segments[img]], axis=0)
                else:
                    h_fg += np.sum(self.imgs_color_histograms[img][fg_segments[img]], axis=0)
                if h_bg is None:
                    h_bg = np.sum(self.imgs_color_histograms[img][bg_segments[img]], axis=0)
                else:
                    h_bg += np.sum(self.imgs_color_histograms[img][bg_segments[img]], axis=0)
        fg_cumulative_hist = h_fg / h_fg.sum()
        bg_cumulative_hist = h_bg / h_bg.sum()

        for img in self.images:
            if img in fg_segments:
                foreground = fg_segments[img]
            else:
                foreground = []
            if img in bg_segments:
                background = bg_segments[img]
            else:
                background = []
            graph_cut = coseg.do_graph_cut((fg_cumulative_hist, bg_cumulative_hist),
                                           (foreground, background),
                                           self.imgs_color_histograms_normalized[img],
                                           self.imgs_segment_neighbors[img])

            segmentation = coseg.pixels_for_segment_selection(self.imgs_segmented[img], np.nonzero(graph_cut))

            self.imgs_cosegmented[img] = np.uint8(segmentation * 255)

    # write the segmented images to specified folder
    def save_segmented_images(self, folder):
        for image in self.imgs_segmented:
            slic.save_superpixel_image(self.imgs_float64[image], self.imgs_segmented[image],
                                       folder + '/' + image.split('/')[-1])

    def plot_cosegmentations(self):
        for img in self.images:
            plt.subplot(1, 2, 2), plt.xticks([]), plt.yticks([])
            plt.title('segmentation')
            plt.imshow(self.imgs_cosegmented[img])

            plt.subplot(1, 2, 1), plt.xticks([]), plt.yticks([])
            superpixels = mark_boundaries(self.imgs_float64[img], self.imgs_segmented[img])
            marking = cv2.imread('markings/' + img.split('/')[-1])
            if marking is not None:
                superpixels[marking[:, :, 0] != 255] = (1, 0, 0)
                superpixels[marking[:, :, 2] != 255] = (0, 0, 1)
            plt.imshow(superpixels)
            plt.title("Superpixels + markings")
            
            plt.savefig("output/segmentation/" + img.split('/')[-1], bbox_inches='tight', dpi=96)

    # Function that uses marking images in the markings folder to construct fg_segments and bg_segments
    # - Marking images should be the same filename as the image they mark
    # - Marking images should be white with red pixels marking foreground and blue pixels marking background
    def get_fg_bg_from_markings(self):
        fg_segments = dict()
        bg_segments = dict()
        for img_path in self.images:
            marking = cv2.imread('markings/'+img_path.split('/')[-1])
            if marking is not None:
                fg_segments[img_path] = np.unique(self.imgs_segmented[img_path][marking[:, :, 0] != 255])
                bg_segments[img_path] = np.unique(self.imgs_segmented[img_path][marking[:, :, 2] != 255])
        return fg_segments, bg_segments


def main():
    image_paths = ['images/bear1.jpg', 'images/bear2.jpg', 'images/bear3.jpg', 'images/bear4.jpg', 'images/bear5.jpg']

    alg = Algorithms(image_paths)
    alg.slic(300)
    alg.save_segmented_images('output/superpixel')

    fg_segments, bg_segments = alg.get_fg_bg_from_markings()

    alg.perform_cosegmentation(fg_segments, bg_segments)

    alg.plot_cosegmentations()


if __name__ == '__main__':
    main()

# TODO
# SLIC superpixel labels containing pixels could be more efficient with numpy
# Work on histograms

