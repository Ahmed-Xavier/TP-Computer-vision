import cv2
import numpy as np


def order_corners(points):
    """Order points as TL, TR, BR, BL."""
    points = np.asarray(points, dtype=np.float32)

    center = np.mean(points, axis=0)

    angles = np.arctan2(
        points[:, 1] - center[1],
        points[:, 0] - center[0],
    )

    points = points[np.argsort(angles)]

    sums = points[:, 0] + points[:, 1]
    start = np.argmin(sums)

    points = np.roll(points, -start, axis=0)

    return points.astype(np.float32)


def find_page_corners(image, params):
    """
    Detect exactly four paper corners using the binary image geometry.

    No:
        - Canny
        - contours
        - Harris
        - SIFT
        - Hough lines

    The document is assumed to already be black and white.
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # --------------------------------------------------
    # Convert image to binary
    # --------------------------------------------------

    _, binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    # --------------------------------------------------
    # Calculate the amount of white pixels on every row
    # and every column.
    # --------------------------------------------------

    white = binary == 255

    row_ratio = np.mean(white, axis=1)
    col_ratio = np.mean(white, axis=0)

    # --------------------------------------------------
    # Find the largest continuous white region touching
    # the image interior.
    #
    # The paper should occupy a significant portion of
    # the image.
    # --------------------------------------------------

    row_threshold = params.get(
        "row_white_ratio",
        0.50,
    )

    col_threshold = params.get(
        "column_white_ratio",
        0.50,
    )

    rows = np.where(row_ratio >= row_threshold)[0]
    cols = np.where(col_ratio >= col_threshold)[0]

    if len(rows) < 2 or len(cols) < 2:
        raise RuntimeError(
            "Could not find a sufficiently large black/white page region."
        )

    # --------------------------------------------------
    # Initial bounding rectangle.
    # --------------------------------------------------

    top = int(rows[0])
    bottom = int(rows[-1])

    left = int(cols[0])
    right = int(cols[-1])

    if bottom <= top or right <= left:
        raise RuntimeError(
            "Invalid page bounds detected."
        )

    # --------------------------------------------------
    # Refine each corner locally.
    #
    # Search for the strongest transition around the
    # four initial corners.
    # --------------------------------------------------

    search = int(
        params.get(
            "corner_search_radius",
            40,
        )
    )

    def refine_corner(x0, y0):
        x_min = max(0, x0 - search)
        x_max = min(w - 1, x0 + search)

        y_min = max(0, y0 - search)
        y_max = min(h - 1, y0 + search)

        best_point = (x0, y0)
        best_score = -1

        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):

                # Local horizontal transition.
                if 1 <= x < w - 1:
                    dx = abs(
                        int(gray[y, x + 1])
                        - int(gray[y, x - 1])
                    )
                else:
                    dx = 0

                # Local vertical transition.
                if 1 <= y < h - 1:
                    dy = abs(
                        int(gray[y + 1, x])
                        - int(gray[y - 1, x])
                    )
                else:
                    dy = 0

                score = dx + dy

                if score > best_score:
                    best_score = score
                    best_point = (x, y)

        return np.array(
            best_point,
            dtype=np.float32,
        )

    corners = np.array(
        [
            refine_corner(left, top),
            refine_corner(right, top),
            refine_corner(right, bottom),
            refine_corner(left, bottom),
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------
    # Validate
    # --------------------------------------------------

    margin = 5

    for x, y in corners:

        if not (
            margin <= x < w - margin
            and
            margin <= y < h - margin
        ):
            raise RuntimeError(
                "Detected page corner lies outside the image."
            )

    # Make sure the quadrilateral is meaningful.
    area = abs(
        cv2.contourArea(
            corners.reshape((-1, 1, 2))
        )
    )

    if area < 0.20 * w * h:
        raise RuntimeError(
            f"Detected page is too small: "
            f"{area / (w * h):.2%} of image."
        )

    return order_corners(corners)