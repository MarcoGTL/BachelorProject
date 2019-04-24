import cv2
import argparse
import numpy


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())

    img = cv2.imread(args["image"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d_SIFT.create()
    kp, des = sift.detectAndCompute(gray,None)

    cv2.drawKeypoints(gray, kp, img)
    cv2.imwrite("sift_keypoints.jpg", img)

    cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints_rich.jpg', img)


if __name__ == '__main__':
    main()

