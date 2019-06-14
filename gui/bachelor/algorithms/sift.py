import cv2

"""
A collection of functions for SIFT keypoints and descriptors, which are currently not used in the pipeline.
"""


def get_sift_keypoints_and_descriptors(image_path, mask=None):
    """ Returns SIFT keypoints and descriptors

    Parameters:
        image_path (str): Relative path to the image
        mask (list of lists): Mask for the input image. Should be uint8.
    Returns:
        keypoints, descriptors
    """
    gray_img_uint8 = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    sift = cv2.xfeatures2d_SIFT.create()
    return sift.detectAndCompute(gray_img_uint8, mask)


# accepts an uint8 input image, keypoints, and uint8 BGR output image, and optional detailed boolean
# outputs the keypoints as images
def draw_keypoints(image_path, keypoints, detailed=False):
    """ Saves an image with keypoints drawn on a copy of the input image.

    Parameters:
        image_path (str): Relative path to the image
        keypoints: List of keypoints obtained using get_sift_keypoints_and_descriptors
        detailed (bool): Boolean indicating whether detailed keypoints should be drawn
    """
    gray_img_uint8 = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    out_img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
    if detailed:
        cv2.drawKeypoints(gray_img_uint8, keypoints, out_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite('sift_keypoints_rich.png', out_img)
    else:
        cv2.drawKeypoints(gray_img_uint8, keypoints, out_img)
        cv2.imwrite("sift_keypoints.png", out_img)
