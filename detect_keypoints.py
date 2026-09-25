import cv2


def detect_sift_features(image, params: dict):
    """Detect SIFT keypoints and descriptors."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(
        nfeatures=params.get("nfeatures", 500)
    )

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors