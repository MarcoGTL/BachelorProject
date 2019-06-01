import cv2
import numpy as np
from scipy.stats import entropy


# Calculates a color feature vector consisting of means of B, G, R, H, S, V,
# and a histogram and their entropy of Hue (5 bins) and Saturation (3bins)
def get_color_feature(image_bgr, mask):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.uint8(mask)

    means_BGR = np.asarray(cv2.mean(image_bgr, mask)[0:3])
    means_HSV = np.asarray(cv2.mean(image_hsv, mask)[0:3])

    hist_H = cv2.calcHist([image_hsv], [0], mask, [5], [0, 181]).flatten()
    entropy_H = entropy(hist_H)

    hist_S = cv2.calcHist([image_hsv], [1], mask, [3], [0, 256]).flatten()
    entropy_S = entropy(hist_S)

    return np.concatenate((means_BGR, means_HSV, hist_H, [entropy_H], hist_S, [entropy_S])).astype('float32')
