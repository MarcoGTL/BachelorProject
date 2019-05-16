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
    def __init__(self):
        self.imgs_float64 = dict()
        self.imgs_segmented = dict()
        self.imgs_segment_ids = dict()
        self.imgs_segment_neighbors = dict()
        self.imgs_color_histograms = dict()
        self.imgs_color_histograms_normalized = dict()
        self.imgs_cosegmented = dict()

    # generate super-pixel segments from images in img_paths using SLIC
    def slic(self, img_paths, num_segments, compactness=10.0, max_item=10, sigma=5):
        for image in img_paths:
            self.imgs_float64[image] = slic.read_image_as_float64(image)
            self.imgs_segmented[image] = slic.get_segmented_image(self.imgs_float64[image],
                                                                  num_segments, compactness, max_item, sigma)
            self.imgs_segment_ids[image] = np.unique(self.imgs_segmented[image])

    def get_segmented_images(self, img_paths):
        segmented_images = dict()
        for image in img_paths:
            segmented_images[image] = self.imgs_segmented[image]
        return segmented_images

    def save_segmented_images(self, img_paths):
        for image in img_paths:
            slic.save_superpixel_image(self.imgs_float64[image], self.imgs_segmented[image],
                                       'output/superpixel/'+image.split('/')[-1])

    def get_color_histogram(self, img_path, segment_id):
        return self.imgs_color_histograms[img_path][segment_id]

    def perform_cosegmentations(self, img_paths, fg_segments, bg_segments):
        for img_path in img_paths:
            centers = np.array([np.mean(np.nonzero(self.imgs_segmented[img_path] == i), axis=1)
                                for i in self.imgs_segment_ids[img_path]])

            self.imgs_segment_neighbors[img_path] = Delaunay(centers).vertex_neighbor_vertices

            hsv = cv2.cvtColor(self.imgs_float64[img_path].astype('float32'), cv2.COLOR_BGR2HSV)
            bins = [20, 20]  # H = S = 20
            ranges = [0, 360, 0, 1]  # H: [0, 360], S: [0, 1]
            self.imgs_color_histograms[img_path] = np.float32(
                [cv2.calcHist([hsv], [0, 1], np.uint8(self.imgs_segmented[img_path] == i), bins, ranges).flatten()
                 for i in self.imgs_segment_ids[img_path]])
            self.imgs_color_histograms_normalized[img_path] = coseg.normalize_histograms(
                self.imgs_color_histograms[img_path])

        # get cumulative BG/FG histograms, being the sum of the selected superpixel IDs and then normalized
        h_fg = None
        h_bg = None
        for img_path in img_paths:
            if img_path in fg_segments:
                if h_fg is None:
                    h_fg = np.sum(self.imgs_color_histograms[img_path][fg_segments[img_path]], axis=0)
                else:
                    h_fg += np.sum(self.imgs_color_histograms[img_path][fg_segments[img_path]], axis=0)
                if h_bg is None:
                    h_bg = np.sum(self.imgs_color_histograms[img_path][bg_segments[img_path]], axis=0)
                else:
                    h_bg += np.sum(self.imgs_color_histograms[img_path][bg_segments[img_path]], axis=0)
        fg_cumulative_hist = h_fg / h_fg.sum()
        bg_cumulative_hist = h_bg / h_bg.sum()

        for img_path in img_paths:
            if img_path in fg_segments:
                foreground = fg_segments[img_path]
            else:
                foreground = []
            if img_path in bg_segments:
                background = bg_segments[img_path]
            else:
                background = []
            graph_cut = coseg.do_graph_cut((fg_cumulative_hist, bg_cumulative_hist),
                                           (foreground, background),
                                           self.imgs_color_histograms_normalized[img_path],
                                           self.imgs_segment_neighbors[img_path])

            segmentation = coseg.pixels_for_segment_selection(self.imgs_segmented[img_path], np.nonzero(graph_cut))

            self.imgs_cosegmented[img_path] = np.uint8(segmentation * 255)

    def get_cosegmented_images(self, img_paths):
        cosegmented_images = dict()
        for image in img_paths:
            cosegmented_images[image] = self.imgs_cosegmented[image]
        return cosegmented_images

    # Function that uses marking images in the markings folder to construct fg_segments and bg_segments
    # - Marking images should be the same filename as the image they mark
    # - Marking images should be white with red pixels marking foreground and blue pixels marking background
    def get_fg_bg_from_markings(self, img_paths):
        fg_segments = dict()
        bg_segments = dict()
        for img_path in img_paths:
            marking = cv2.imread('markings/'+img_path.split('/')[-1])
            if marking is not None:
                fg_segments[img_path] = np.unique(self.imgs_segmented[img_path][marking[:, :, 0] != 255])
                bg_segments[img_path] = np.unique(self.imgs_segmented[img_path][marking[:, :, 2] != 255])
        return fg_segments, bg_segments

    def plot_cosegmentations(self, img_paths):
        for img_path in img_paths:
            plt.subplot(1, 2, 2), plt.xticks([]), plt.yticks([])
            plt.title('segmentation')
            cv2.imwrite("output/segmentation/mask_"+img_path.split('/')[-1], self.imgs_cosegmented[img_path])
            plt.imshow(self.imgs_cosegmented[img_path])
            plt.subplot(1, 2, 1), plt.xticks([]), plt.yticks([])
            img = mark_boundaries(self.imgs_float64[img_path], self.imgs_segmented[img_path])
            marking = cv2.imread('markings/' + img_path.split('/')[-1])
            if marking is not None:
                img[marking[:, :, 0] != 255] = (1, 0, 0)
                img[marking[:, :, 2] != 255] = (0, 0, 1)
            plt.imshow(img)
            plt.title("SLIC + markings")
            plt.savefig("output/segmentation/"+img_path.split('/')[-1], bbox_inches='tight', dpi=96)


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image", required=True, help="Path to the image")
    # parser.add_argument("--segments", type=int, default=300, help="Number of segments")
    # parser.add_argument("--save", default=False, help="Save the image?")
    # parser.add_argument("--labels", nargs='+', type=int,
    #                    help="Segment labels marking region where features should be detected")
    # args = vars(parser.parse_args())

    image_paths = ['images/bear1.jpg', 'images/bear2.jpg', 'images/bear3.jpg', 'images/bear4.jpg', 'images/bear5.jpg']

    alg = Algorithms()
    alg.slic(image_paths, 300)
    alg.get_segmented_images(image_paths)
    alg.save_segmented_images(image_paths)

    fg_segments, bg_segments = alg.get_fg_bg_from_markings(image_paths)

    alg.perform_cosegmentations(image_paths, fg_segments, bg_segments)

    alg.plot_cosegmentations(image_paths)

    hist = alg.get_color_histogram('images/bear2.jpg', 25)


if __name__ == '__main__':
    main()

# TODO
# SLIC superpixel labels containing pixels could be more efficient with numpy
# Work on histograms

