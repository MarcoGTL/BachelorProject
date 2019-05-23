import cv2


# accepts path to a BGR image and returns its uint8 grayvalue representation to be used for sift
def read_image_as_uint8(filename):
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)


# accepts an uint8 grayvalue image and returns the SIFT keypoints and descriptors
def get_sift_keypoints_and_descriptors(gray_img_uint8, mask=None):
    sift = cv2.xfeatures2d_SIFT.create()
    return sift.detectAndCompute(gray_img_uint8, mask)


# accepts an uint8 input image, keypoints, and uint8 BGR output image, and optional detailed boolean
# outputs the keypoints as images
def draw_keypoints(img_uint8, keypoints, out_img_uint8, detailed=False):
    if detailed:
        cv2.drawKeypoints(img_uint8, keypoints, out_img_uint8, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite('sift_keypoints_rich.png', out_img_uint8)
    else:
        cv2.drawKeypoints(img_uint8, keypoints, out_img_uint8)
        cv2.imwrite("sift_keypoints.png", out_img_uint8)
