import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASS = os.getenv("MQTT_PASS")
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "topst-receiver")
MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", "60"))
MQTT_QOS = int(os.getenv("MQTT_QOS", "1"))

DEVICE_NAMESPACE = os.getenv("DEVICE_NAMESPACE", "topst")  # topst/{device}/...

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR = Path(os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR))).absolute()
AUDIO_DIR = DATA_DIR / "audio"
IMAGE_DIR = DATA_DIR / "images"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_SAVE_INTERVAL = int(os.getenv("AUDIO_SAVE_INTERVAL", "5"))  # seconds

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG/INFO/WARNING/ERROR

_save_env = os.getenv("SAVE_CAMERA_FILES", "true").strip().lower()
if _save_env not in {"true", "false"}:
    raise ValueError("SAVE_CAMERA_FILES must be exactly 'true' or 'false'")
SAVE_CAMERA_FILES = _save_env == "true"
