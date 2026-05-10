"""
AI 모델 확장 설정
Detection + Tracking + Face Recognition 조합을 여기서 선택
"""

# ===== 감지 모델 선택 =====
# 옵션: "yolov8n", "yolov11n", "yolov5n", "rtdetr", "effdetlite"
DETECTOR = "yolov8n"

# ===== 추적 모델 선택 =====
# 옵션: "bytetrack", "deepsort"
TRACKER = "bytetrack"

# ===== 모델 파일 경로 =====
YOLO11N_PATH = "yolov11n.pt"
YOLOV8N_PATH = "yolov8n.pt"
YOLOV5N_PATH = "yolov5n.pt"  

# ===== 추론 설정 =====
CONFIDENCE_THRESHOLD = 0.5  # 감지 임계값
IOU_THRESHOLD = 0.45        # NMS IOU 임계값
MAX_DETECTIONS = 100        # 최대 감지 수

# ===== 추적 설정 =====
TRACK_BUFFER = 30           # 추적 버퍼 (프레임)
TRACK_THRESH = 0.5          # 추적 임계값

# ===== 벤치마크 설정 =====
BENCHMARK_MODE = True      # True면 FPS 측정
BENCHMARK_FRAMES = 100      # 벤치마크 프레임 수

# ===== Face Matcher 설정 =====
# 옵션: "simple"
FACE_MATCHER = "simple"

# ===== Embedding L2 정규화 설정 =====
EMBED_L2_NORM = True

# ===== Cosine similarity threshold ===== 
#   - 0.55 ~ 0.65: 널널한 매칭
#   - 0.65 ~ 0.75: 적절 (추천)
#   - 0.80 이상: 매우 보수적인 매칭
MATCH_THRESHOLD = 0.65

# ====== 저장된 사용자 DB 위치 =====
FACE_DB_PATH = "faces/face_db.json"

# ===== (Face Detect) 모델 파일 경로 =====
ARCFACE_ONNX_PATH = "arcface_r100.onnx"

# ===== 얼굴 최소 크기 (pixel * pixel) ===== 
FACE_MIN_AREA = 80 * 80


# ===== Face recognition retry policy =====
FACE_RETRY_INTERVAL_SEC = 0.3   # n_sec
FACE_MAX_TRY = 7                # m_try
FACE_MIN_SIZE = 80         

