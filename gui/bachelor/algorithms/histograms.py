import cv2
from matplotlib import pyplot as plt
import numpy as np


def get_b_histogram(image_bgr, mask=None, bins_b=256):
    if mask is not None:
        mask = np.uint8(mask)
    return cv2.calcHist([image_bgr], [0], mask, [bins_b], [0, 256])


def get_g_histogram(image_bgr, mask=None, bins_g=256):
    if mask is not None:
        mask = np.uint8(mask)
    return cv2.calcHist([image_bgr], [1], mask, [bins_g], [0, 256])


def get_r_histogram(image_bgr, mask=None, bins_r=256):
    if mask is not None:
        mask = np.uint8(mask)
    return cv2.calcHist([image_bgr], [2], mask, [bins_r], [0, 256])


def get_bgr_histogram(image_bgr, mask=None, bins_b=256, bins_g=256, bins_r=256):
    if mask is not None:
        mask = np.uint8(mask)
    return cv2.calcHist([image_bgr], [0, 1, 2], mask, [bins_b, bins_g, bins_r], [0, 256, 0, 256, 0, 256])


def get_hs_histogram(image_bgr, mask=None, bins_h=181, bins_s=256):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if mask is not None:
        mask = np.uint8(mask)
    return cv2.calcHist([image_hsv], [0, 1], mask, [bins_h, bins_s], [0, 181, 0, 256])


# Accepts a list of one-channeled histograms and plots them in one figure
def plot_histograms(histograms):
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')
    for i, histogram in enumerate(histograms):
        plt.plot(histogram)
    plt.show()


# Accepts one two-channeled hue, saturation histogram and plots it
def plot_hs_histogram(histogram_hs):
    plt.imshow(histogram_hs, interpolation='nearest')
    plt.xlabel('Saturation bins')
    plt.ylabel('Hue bins')
    plt.show()
