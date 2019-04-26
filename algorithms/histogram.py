import cv2
from matplotlib import pyplot as plt


# accepts an uint8 image and returns a dictionary of histograms with their color channels as keys
def get_histogram(img_uint8, mask=None, colors=('b', 'g', 'r')):
    histogram = {}
    for i, color in enumerate(colors):
        histogram[color] = cv2.calcHist([img_uint8], [i], mask, [256], [0, 256])
    return histogram


# Accepts a dictionary of histograms with their colors as keys and plot them in one figure
def plot_histogram(histogram):
    channels = histogram.keys()
    title = 'Histogram of ' + len(channels).__str__() + ' color channel(s) ' + ''.join(channels)
    plt.figure(num=title)
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')
    for i, color in enumerate(channels):
        plt.plot(histogram[color], color=color)
        plt.xlim([0, 256])
    plt.show()
