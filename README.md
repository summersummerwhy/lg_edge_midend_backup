# AI 모델 최적화 - 모듈화 및 벤치마크

기존 YOLO11n + BoT-SORT를 모듈화하고, 여러 조합을 테스트할 수 있는 구조

---

## 🎯 목표

1. **속도 개선**: 더 빠른 모델로 FPS 향상
2. **정확도 유지**: 사람 감지 + 추적 정확도 유지
3. **유연성**: 모델 조합을 쉽게 교체 가능
4. **벤치마크**: 자동으로 성능 비교

---

## 📊 지원 모델

### 감지 모델 (Detection)
- **YOLO11n** (현재) - 정확하고 빠름
- **YOLOv8n** - YOLO11n과 비슷, 약간 더 빠름
- **MobileNet-SSD** - 가장 빠름, 정확도 낮음

### 추적 모델 (Tracking)
- **BoT-SORT** (현재) - 가장 정확, 느림
- **ByteTrack** - BoT-SORT보다 2배 빠름, 정확도 비슷
- **DeepSORT** - 중간 성능

---

## 🚀 사용 방법

### 1️⃣ 모델 선택

`app/ai/ai_config.py` 파일 수정:

```python
# 감지 모델 선택
DETECTOR = "yolov8n"  # yolo11n / yolov8n / mobilenet_ssd

# 추적 모델 선택
TRACKER = "bytetrack"  # botsort / bytetrack / deepsort
```

### 2️⃣ 기존 코드 그대로 사용

```python
from app.ai.main import track_image_by_path, track_image

# 파일 경로로 추적
payloads = track_image_by_path(Path("image.jpg"))

# 이미지 배열로 추적
image = cv2.imread("image.jpg")
payloads = track_image(image, "jpg")
```

**기존 코드 수정 없이 바로 사용 가능!** ✅

### 3️⃣ 벤치마크 실행

```bash
# 모든 조합 테스트
python benchmark/benchmark.py --frames 100

# 특정 조합만 테스트
python benchmark/benchmark.py --detector yolov8n --tracker bytetrack --frames 50
```

---

## 📈 예상 성능

### FPS 비교 (예상)

| Detector | Tracker | 예상 FPS | 정확도 |
|----------|---------|----------|--------|
| YOLO11n | BoT-SORT | 10-15 | ⭐⭐⭐⭐⭐ |
| **YOLOv8n** | **ByteTrack** | **20-25** ⭐ | ⭐⭐⭐⭐ |
| YOLOv8n | DeepSORT | 18-22 | ⭐⭐⭐⭐ |
| MobileNet-SSD | ByteTrack | 25-30 | ⭐⭐⭐ |


---


## 📊 벤치마크 결과 예시

```
📊 BENCHMARK SUMMARY
================================================================================

Detector        Tracker         Avg FPS    Min FPS    Max FPS
----------------------------------------------------------------------
yolov8n         bytetrack         24.32      18.20      32.10
  ⭐ FASTEST!
yolov8n         deepsort          21.45      16.80      28.30
yolo11n         bytetrack         18.67      14.20      24.50
yolo11n         botsort           12.34       9.10      16.20
```

---

## 🎯 다음 단계

### Phase 1: 속도 개선 (현재)
- [x] 모듈화
- [x] 여러 모델 추가
- [x] 벤치마크 도구
- [x] 실제 TOPST에서 테스트

### Phase 2: 정확도 개선 (나중)
- [x] 얼굴 인식 추가 (MobileFaceNet)
- [ ] 세부 사물 인식 (EfficientDet-Lite)

### Phase 3: 최적화
- [ ] 모델 경량화 (양자화)
- [ ] 병렬 처리 (멀티 코어 활용)
- [ ] 프레임 스킵 전략

---
