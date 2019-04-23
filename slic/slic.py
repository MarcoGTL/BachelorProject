from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import numpy


# accepts path to the image and returns a float representation of the image
# removes transparency if needed
def get_image_as_float(filename):
    image = img_as_float(io.imread(filename))
    if len(image[0][0]) > 3:
        image = numpy.compress([True, True, True], image, axis=2)
    return image


# accepts float representation of image and number of (approximate) segments
# returns a 2d list of pixels of the image containing their segment labels from SLIC
# for optional arguments refer to https://scikit-image.org/docs/dev/api/skimage.segmentation.html?highlight=slic
def get_pixel_segments(image, number_of_segments, compactness=10.0, max_iter=10, sigma=0):
    pixels = slic(image, n_segments=number_of_segments, compactness=compactness, max_iter=max_iter, sigma=sigma)
    return pixels


# accepts 2d list of pixels containing their segment labels
# returns 1d list of segments containing their pixels
def get_segment_pixels(pixel_segments):
    # calculate pixels for each segment
    segments_pixels = [[] for i in range(len(pixel_segments*pixel_segments))]
    for i in range(len(pixel_segments)):
        for j in range(len(pixel_segments)):
            segments_pixels[pixel_segments[i][j]].append(tuple((i, j)))
    print(segments_pixels[0])
    return segments_pixels


def show_plot(image, pixel_segments):
    fig = plt.figure("Superpixels")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, pixel_segments))
    plt.axis("off")
    plt.show()


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-s", "--segments", type=int, default=300, help="Number of segments")
    ap.add_argument("--save", type=bool, default=False, help="Save the image?")
    args = vars(ap.parse_args())

    image = get_image_as_float(args["image"])

    segments = get_pixel_segments(image, args["segments"])

    if args["save"]:
        io.imsave('super-pixels.png', mark_boundaries(image, segments))

    show_plot(image, segments)


if __name__ == '__main__':
    main()
