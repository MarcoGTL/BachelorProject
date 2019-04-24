import cv2
import slic


def main():
    parser = slic.argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the image")
    parser.add_argument("--segments", type=int, default=300, help="Number of segments")
    parser.add_argument("--save", help="Save the image?")
    parser.add_argument("--labels", nargs='+', type=int,
                        help="Segment labels marking region where features should be detected")
    args = vars(parser.parse_args())

    # generate super-pixel segments from image using SLIC
    img_float64 = slic.get_image_as_float(args["image"])
    segments = slic.get_pixel_segments(img_float64, args["segments"])

    # save super-pixel representation or plot it
    if args["save"]:
        slic.io.imsave('super-pixels.png', slic.mark_boundaries(img_float64, segments))
    else:
        slic.show_plot(img_float64, segments)

    img_uint8 = cv2.imread(args["image"])
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d_SIFT.create()

    # detect features at specified super-pixel labels, or over entire image
    if args["labels"]:
        mask = slic.get_mask(segments, args["labels"])
        slic.io.imsave('mask.png', mask)
        kp, des = sift.detectAndCompute(gray, mask)
    else:
        kp, des = sift.detectAndCompute(gray, None)

    # draw keypoints
    cv2.drawKeypoints(gray, kp, img_uint8)
    cv2.imwrite("sift_keypoints.jpg", img_uint8)

    # draw more detailed keypoints
    cv2.drawKeypoints(gray, kp, img_uint8, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints_rich.jpg', img_uint8)


if __name__ == '__main__':
    main()

