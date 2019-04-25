import cv2
from matplotlib import pyplot as plt


def get_histogram(img_uint8, mask=None, colors=('b', 'g', 'r')):
    histogram = {}
    for i, color in enumerate(colors):
        histogram[color] = cv2.calcHist([img_uint8], [i], mask, [256], [0, 256])
    return histogram


def plot_histogram(histogram, colors=('b', 'g', 'r')):
    print(histogram)
    for i, color in enumerate(colors):
        plt.plot(histogram[color], color=color)
        plt.xlim([0, 256])
    plt.show()


def histograms(img_uint8, mask=None):
    channels = cv2.split(img_uint8)
    colors = ('b', 'g', 'r')
    plt.figure()
    plt.title('Color Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')
    features = []

    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], mask, [256], [0, 256])
        features.extend(hist)

        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.show()
