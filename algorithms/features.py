import cv2
import numpy as np
from scipy.stats import entropy


# Calculates a feature vector consisting of means of B, G, R, H, S, V,
# and a histogram and their entropy of Hue (5 bins) and Saturation (3bins)
def get_color_feature(image_bgr, mask, bins_h=5, bins_s=3):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.uint8(mask)

    means_BGR = np.asarray(cv2.mean(image_bgr, mask)[0:3])
    means_HSV = np.asarray(cv2.mean(image_hsv, mask)[0:3])

    hist_H = cv2.calcHist([image_hsv], [0], mask, [bins_h], [0, 181]).flatten()
    entropy_H = entropy(hist_H)

    hist_S = cv2.calcHist([image_hsv], [1], mask, [bins_s], [0, 256]).flatten()
    entropy_S = entropy(hist_S)

    return np.concatenate((means_BGR, means_HSV, hist_H, [entropy_H], hist_S, [entropy_S])).astype('float32')


# Calculates a feature vector from a normalized hsv histogram of bins_h and bins_s
def get_hsv_histogram_feature(image_bgr, mask, bins_h=20, bins_s=20):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([image_hsv], [0, 1], np.uint8(mask), [bins_h, bins_s], [0, 181, 0, 256]).flatten()
    return np.float32(histogram / histogram.sum())


# Calculates a feature vector from the sift descriptor obtained from a keypoint at location (kp_x, kp_y)
# and size kp_size, with a fixed angle of -1
def get_sift_feature(image_bgr, kp_x, kp_y, kp_size=32.0):
    sift = cv2.xfeatures2d_SIFT.create()
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kp = cv2.KeyPoint(kp_x, kp_y, kp_size, -1)
    kp, desc = sift.compute(image_gray, [kp])
    return np.float32(desc[0])

