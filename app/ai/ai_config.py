"""
AI 모델 설정
모델 조합을 여기서 선택
"""

# ===== 감지 모델 선택 =====
# 옵션: "yolov8n", "yolo11n", "yolo_nano", "mobilenet_ssd"
DETECTOR = "mobilenet_ssd"

# ===== 추적 모델 선택 =====
# 옵션: "bytetrack", "botsort", "deepsort"
TRACKER = "bytetrack"

# ===== 모델 파일 경로 =====
YOLO11N_PATH = "yolo11n.pt"
YOLOV8N_PATH = "yolov8n.pt"

# ===== 추론 설정 =====
CONFIDENCE_THRESHOLD = 0.5  # 감지 임계값
IOU_THRESHOLD = 0.45        # NMS IOU 임계값
MAX_DETECTIONS = 100        # 최대 감지 수

# ===== 추적 설정 =====
TRACK_BUFFER = 30           # 추적 버퍼 (프레임)
TRACK_THRESH = 0.5          # 추적 임계값
MATCH_THRESH = 0.8          # 매칭 임계값

# ===== 벤치마크 설정 =====
BENCHMARK_MODE = True      # True면 FPS 측정
BENCHMARK_FRAMES = 100      # 벤치마크 프레임 수
