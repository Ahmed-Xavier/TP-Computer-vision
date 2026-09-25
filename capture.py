import cv2
import yaml


def get_frame(params: dict):
    """Grab one frame from a webcam index or an image/video file, resized to a fixed width.

    For a webcam source, shows a live preview first — press SPACE to capture, ESC to cancel.
    """
    source = params["source"]
    resize_width = params["resize_width"]

    # Webcam index (0, 1, 2...) vs file path
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {source}")

        frame = None
        while True:
            ok, live = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"Could not read from webcam index {source}")

            cv2.imshow("Live feed - SPACE to capture, ESC to cancel", live)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # SPACE
                frame = live
                break
            elif key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                raise RuntimeError("Capture cancelled by user")

        cap.release()
        cv2.destroyAllWindows()
    else:
        frame = cv2.imread(source)
        if frame is None:
            # Not a still image -> try as video, grab first frame
            cap = cv2.VideoCapture(source)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"Could not read from source: {source}")

    # Resize, keeping aspect ratio
    h, w = frame.shape[:2]
    new_h = int(h * (resize_width / w))
    frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

    return frame
