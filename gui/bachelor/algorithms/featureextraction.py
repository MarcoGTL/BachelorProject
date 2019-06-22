import cv2
import numpy as np
from scipy.stats import entropy


class FeatureExtraction:
    """
    A class containing methods to extract features from an image

    Args:
        image_path (str): Relative path to the image

    Attributes:
        image_bgr (ndarray): A 3D array representation of the image of shape (row, column, pixel), where each pixel is
            represented by three numpy.uint8 values corresponding to BGR.
        image_hsv (ndarray): A 3D array representation of the image of shape (row, column, pixel), where each pixel is
            represented by three numpy.uint8 values corresponding to HSV.
        image_gray (ndarray): A 2D array representation of the image of shape (row, column), where each entry is
            a numpy.uint8 grayvalue of the pixel.
    """
    def __init__(self, image_path: str):
        print(image_path)
        self.image_bgr = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        self.image_hsv = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2HSV)
        self.image_gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)

    def means_bgr(self, mask):
        """ List of means BGR

        Parameters:
            mask (ndarray): 2D array of uint8 values with same dimension as the image,
                where 0 entries indicate pixels to exclude.

        Returns: List of the means of B, G, and R
        """
        means = cv2.mean(self.image_bgr, mask)[0:3]
        return means

    def means_hsv(self, mask):
        """ List of means HSV

        Parameters:
            mask (ndarray): 2D array of uint8 values with same dimension as the image,
                where 0 entries indicate pixels to exclude.

        Returns: List of the means of H, S, and V
        """
        means = cv2.mean(self.image_hsv, mask)[0:3]
        return means

    def h_hist(self, mask, bins, include_entropy=False):
        """ Hue Histogram

        Parameters:
            mask (ndarray): 2D array of uint8 values with same dimension as the image,
                where 0 entries indicate pixels to exclude.
            bins (int): The number of bins
            include_entropy (bool): Indicates if the entropy should be appended to the output

        Returns: List of the histogram
        """
        histogram = cv2.calcHist([self.image_hsv], [0], mask, [bins], [0, 181]).flatten().tolist()
        if include_entropy:
            histogram += [entropy(histogram)]
        return histogram

    def s_hist(self, mask, bins, include_entropy=False):
        """ Saturation Histogram

        Parameters:
            mask (ndarray): 2D array of uint8 values with same dimension as the image,
                where 0 entries indicate pixels to exclude.
            bins (int): The number of bins
            include_entropy (bool): Indicates if the entropy should be appended to the output

        Returns: List of the histogram
        """
        histogram = cv2.calcHist([self.image_hsv], [1], mask, [bins], [0, 256]).flatten().tolist()
        if include_entropy:
            histogram += [entropy(histogram)]
        return histogram

    def hs_hist(self, mask, bins_h=5, bins_s=3):
        """ 2D Hue/Saturation Histogram

        Parameters:
            mask (ndarray): 2D array of uint8 values with same dimension as the image,
                where 0 entries indicate pixels to exclude.
            bins_h (int): The number of bins for Hue
            bins_s (int): The number of bins for Saturation

        Returns: List of the flattened histogram
        """
        return cv2.calcHist([self.image_hsv], [0, 1], np.uint8(mask),
                            [bins_h, bins_s], [0, 181, 0, 256]).flatten().tolist()

    def sift(self, coord_x, coord_y, kp_size=32.0):
        """ Calculates a SIFT descriptor from a keypoint with given location and size, but with fixed angle

        Parameters:
            coord_x (float): x coordinate of the keypoint
            coord_y (float): y coordiante of the keypoint
            kp_size (float): Diameter of the keypoint neighborhood

        Returns: A SIFT descriptor as list of length 128
        """
        sift = cv2.xfeatures2d_SIFT.create()
        kp = cv2.KeyPoint(x=coord_x, y=coord_y, _size=kp_size, _angle=-1)
        kp, desc = sift.compute(self.image_gray, [kp])
        return desc[0].tolist()

    def hog(self, coord_x, coord_y, winSize=(32, 32), blockSize=(16, 16), blockStride=(8, 8),
            cellSize=(8, 8), bins=9):
        """ Calculates a HOG descriptor around a point at coord_x, coord_y.

            coord_x (float): x coordinate of the center
            coord_y (float): y coordinate of the center
            winSize ((int, int)): Size in pixels of the detection window in (width, height) of the HOG descriptor
            blockSize ((int, int)): Block size of the HOG descriptor in pixels. Align to cell size
            blockStride ((int, int)): Block stride of the HOG descriptor. It must be a multiple of cell size
            cellSize ((int, int)): Cell size of the HOG descriptor
            bins (int): Number of bins per cell of the HOG descriptor

        Returns: A HOG descriptor as list of length width * height * 3, where winSize=(width, height)
        """
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, bins)
        hist = hog.compute(self.image_bgr, locations=[(coord_x-winSize[0], coord_y-winSize[1])])
        return hist.flatten().tolist()
