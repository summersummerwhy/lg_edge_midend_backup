import cv2
import logging
from datetime import datetime

from app.config import IMAGE_DIR

log = logging.getLogger(__name__)


def add_new_face(img):
    save_dir = IMAGE_DIR / "rpi4"

    try:
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = save_dir / f"face_{timestamp}.jpg"

        # Save the image
        success = cv2.imwrite(str(filepath), img)

        if success:
            log.info(f"Saved face image to {filepath}")
        else:
            log.error(f"Failed to save face image to {filepath}")

    except Exception as e:
        log.exception(f"Error saving face image: {e}")
