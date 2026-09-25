import os
import cv2

OUTPUT_DIR = "results/"
TIMESTAMP_FORMAT = "%H-%M-%S"


def save_snapshot(image, stage_name, run_id):
    """Save an image with a consistent name: results/01_harris_keypoints_14-32-07.png"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{OUTPUT_DIR}{stage_name}_{run_id}.png"
    cv2.imwrite(filename, image)
    return filename