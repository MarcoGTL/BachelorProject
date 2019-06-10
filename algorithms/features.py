import cv2
import numpy as np
from scipy.stats import entropy


def get_means_bgr_fv(image_bgr, mask):
    return cv2.mean(image_bgr, mask)[0:3]


def get_means_hsv_fv(image_hsv, mask):
    return cv2.mean(image_hsv, mask)[0:3]


def get_h_hist_fv(image_hsv, mask, bins, include_entropy=False):
    hist_h = cv2.calcHist([image_hsv], [0], mask, [bins], [0, 181]).flatten().tolist()
    if include_entropy:
        return hist_h + [entropy(hist_h)]
    return hist_h


def get_s_hist_fv(image_hsv, mask, bins, include_entropy=False):
    hist_s = cv2.calcHist([image_hsv], [1], mask, [bins], [0, 256]).flatten().tolist()
    if include_entropy:
        return hist_s + [entropy(hist_s)]
    return hist_s


# Calculates a feature vector from a normalized hs histogram of bins_h and bins_s
def get_normalized_hs_hist_fv(image_hsv, mask, bins_h=5, bins_s=3):
    histogram = cv2.calcHist([image_hsv], [0, 1], np.uint8(mask), [bins_h, bins_s], [0, 181, 0, 256]).flatten()
    return (histogram / histogram.sum()).tolist()


# Calculates a feature vector from the sift descriptor obtained from a keypoint at location (kp_x, kp_y)
# and size kp_size, with a fixed angle of -1
def get_sift_fv(image_gray, coord_x, coord_y, kp_size=32.0):
    sift = cv2.xfeatures2d_SIFT.create()
    kp = cv2.KeyPoint(coord_x, coord_y, kp_size, -1)
    kp, desc = sift.compute(image_gray, [kp])
    return desc[0].tolist()


def get_hog_fv(image_bgr, coord_x, coord_y, winSize=(32, 32), blockSize=(16, 16), blockStride=(8, 8),
               cellSize=(8, 8), bins=9):
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, bins)
    hist = hog.compute(image_bgr, locations=[(coord_x-winSize[0], coord_y-winSize[1])])
    return hist.flatten().tolist()
