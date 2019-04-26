import slic
import sift
import histogram
import argparse
import copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the image")
    parser.add_argument("--segments", type=int, default=300, help="Number of segments")
    parser.add_argument("--save", default=False, help="Save the image?")
    parser.add_argument("--labels", nargs='+', type=int,
                        help="Segment labels marking region where features should be detected")
    args = vars(parser.parse_args())

    # generate super-pixel segments from image using SLIC
    img_float64 = slic.read_image_as_float64(args["image"])
    segmented_pixels = slic.get_segmented_pixels(img_float64, args["segments"])

    # save super-pixel representation or plot it
    if args["save"]:
        slic.io.imsave('super-pixels.png', slic.mark_boundaries(img_float64, segmented_pixels))
    else:
        slic.show_plot(img_float64, segmented_pixels)

    img_uint8 = sift.read_image_as_uint8(args["image"])
    gray_uint8 = sift.to_gray_value(img_uint8)

    # Create mask if labels are given, otherwise mask=None
    mask = None
    if args["labels"]:
        mask = slic.get_mask(segmented_pixels, args["labels"])
        slic.io.imsave('mask.png', mask)

    # SIFT
    kp = sift.get_keypoints(gray_uint8, mask)
    out = copy.deepcopy(img_uint8)
    sift.draw_keypoints(img_uint8, kp, out)
    sift.draw_keypoints(img_uint8, kp, out, detailed=True)

    # Histograms
    hist = histogram.get_histogram(img_uint8, mask)
    hist_gray = histogram.get_histogram(gray_uint8, mask, colors='k')
    histogram.plot_histogram(hist)
    histogram.plot_histogram(hist_gray)


if __name__ == '__main__':
    main()
