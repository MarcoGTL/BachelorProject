import cv2


# accepts path to the image file and returns uint8 representation of the image
def read_image_as_uint8(filename):
    return cv2.imread(filename)


# accepts an uint8 BGR image and converts it to grayvalue
def to_gray_value(img_uint8):
    return cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)


# accepts an uint8 grayvalue image and returns the SIFT keypoints
def get_keypoints(gray_img_uint8, mask=None):
    sift = cv2.xfeatures2d_SIFT.create()
    return sift.detect(gray_img_uint8, mask)


# accepts an uint8 grayvalue image and returns the descriptors
def get_descriptors(gray_img_uint8, keypoints):
    sift = cv2.xfeatures2d_SIFT.create()
    return sift.compute(gray_img_uint8, keypoints)


# accepts an uint8 input image, keypoints, and uint8 BGR output image, and optional detailed boolean
# outputs the keypoints as images
def draw_keypoints(img_uint8, keypoints, out_img_uint8, detailed=False):
    if detailed:
        cv2.drawKeypoints(img_uint8, keypoints, out_img_uint8, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite('sift_keypoints_rich.png', out_img_uint8)
    else:
        cv2.drawKeypoints(img_uint8, keypoints, out_img_uint8)
        cv2.imwrite("sift_keypoints.png", out_img_uint8)
