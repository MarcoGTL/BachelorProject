from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot
import numpy


# accepts path to the image file and returns a float64 representation of the image
# removes transparency if needed
def read_image_as_float64(filename):
    image = img_as_float(io.imread(filename))
    if len(image[0][0]) > 3:
        image = numpy.compress([True, True, True], image, axis=2)
    return image


# accepts float representation of image and number of (approximate) segments
# returns a 2d list of pixels of the image containing their segment labels from SLIC
# for optional arguments refer to https://scikit-image.org/docs/dev/api/skimage.segmentation.html?highlight=slic
def get_segmented_pixels(image, number_of_segments, compactness=10.0, max_iter=10, sigma=5):
    segmented_pixels = slic(image, n_segments=number_of_segments,
                            compactness=compactness, max_iter=max_iter, sigma=sigma)
    return segmented_pixels


# accepts 2d list of pixels containing their segment labels
# returns 1d list of segments containing their pixels
def get_segments(segmented_pixels):
    # calculate pixels for each segment
    segments = [[] for i in range(len(segmented_pixels*segmented_pixels))]
    for i in range(len(segmented_pixels)):
        for j in range(len(segmented_pixels)):
            segments[segmented_pixels[i][j]].append(tuple((i, j)))
    return segments


# accepts 2d list of segmented pixels and a list of labels,
# return a 2d mask where pixels are 255 if they are in the labels list or 0 if not
def get_mask(segmented_pixels, segment_labels):
    return numpy.where(numpy.isin(segmented_pixels, segment_labels), 255, 0).astype('uint8')


def show_plot(image, pixel_segments):
    fig = matplotlib.pyplot.figure("Superpixels")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, pixel_segments))
    matplotlib.pyplot.axis("off")
    matplotlib.pyplot.show()
