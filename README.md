# Raspberry Pi 4 중간관리자 서버

FastAPI + ONNX Runtime + MQTT를 사용하는 Raspberry Pi 4 기반 중간관리자 서버입니다.

## 📋 요구사항

- Raspberry Pi 4 (2GB RAM 이상 권장)
- Python 3.9+
- Mosquitto MQTT 브로커

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/2025F-CID1-TeamH/mid_end.git
```

### 2. 초기 설정 실행

```bash
chmod +x setup.sh
./setup.sh
```

이 스크립트는 자동으로 다음을 수행합니다:
- 시스템 패키지 설치
- MQTT 브로커(Mosquitto) 설치 및 시작
- Python 가상환경 생성
- 필요한 패키지 설치
- systemd 서비스 등록

### 3. 서버 실행

#### systemd 서비스로 실행 (권장)

```bash
sudo systemctl start rpi-manager
sudo systemctl status rpi-manager
```

#### 수동 실행

```bash
source venv/bin/activate
python main.py
```

## 📁 프로젝트 구조

```
.
├── main.py              # FastAPI 메인 서버
├── config.yaml          # 설정 파일
├── requirements.txt     # Python 패키지 목록
├── setup.sh            # 초기 설정 스크립트
├── Dockerfile          # Docker 이미지 (선택사항)
├── .env                # 환경변수 (자동 생성)
├── models/             # ONNX 모델 저장 디렉토리
├── logs/               # 로그 파일 디렉토리
└── data/               # 데이터 파일 디렉토리
```

## 🔧 설정

### MQTT 브로커 설정

`config.yaml` 또는 `.env` 파일에서 MQTT 브로커 정보를 수정하세요:

```yaml
mqtt:
  broker: "localhost"
  port: 1883
  username: ""
  password: ""
```

### ONNX 모델 추가

ONNX 모델 파일을 `models/` 디렉토리에 추가하고 `config.yaml`에서 경로를 설정하세요:

```yaml
onnx:
  model_path: "models/model.onnx"
```

## 📡 API 엔드포인트

### 기본 엔드포인트

- `GET /` - 서버 상태 확인
- `GET /health` - 헬스체크 (MQTT, ONNX 상태 포함)

### ONNX 추론

- `POST /inference` - 모델 추론 실행

```json
{
  "data": [1.0, 2.0, 3.0],
  "model_name": "default"
}
```

### MQTT 제어

- `POST /mqtt/publish` - MQTT 메시지 발행
- `GET /mqtt/status` - MQTT 연결 상태 확인

```json
{
  "device_id": "device_001",
  "command": "turn_on",
  "params": {"brightness": 80}
}
```

## 🛠 유용한 명령어

### 서비스 관리

```bash
# 서비스 시작
sudo systemctl start rpi-manager

# 서비스 중지
sudo systemctl stop rpi-manager

# 서비스 재시작
sudo systemctl restart rpi-manager

# 서비스 상태 확인
sudo systemctl status rpi-manager

# 로그 실시간 확인
sudo journalctl -u rpi-manager -f
```

### MQTT 테스트

```bash
# 메시지 구독
mosquitto_sub -t "sensors/#" -v

# 메시지 발행
mosquitto_pub -t "sensors/temp" -m "25.5"
```

## 🐛 트러블슈팅

### MQTT 연결 실패

```bash
# Mosquitto 상태 확인
sudo systemctl status mosquitto

# Mosquitto 재시작
sudo systemctl restart mosquitto
```

### 메모리 부족

Raspberry Pi 4의 메모리가 부족한 경우 ONNX 모델 최적화를 고려하세요:
- 모델 양자화 (quantization)
- 더 작은 모델 사용
- Swap 메모리 증가

### 권한 오류

```bash
# 현재 사용자에게 권한 부여
sudo chown -R $USER:$USER .
```

## 📝 TODO
- [ ] 기본 세팅
- [ ] 라즈베리파이에서 구동시켜보기
- [ ] low-end와의 통신
- [ ] high-end와의 통신
- [ ] ONNX 모델 예제 추가
- [ ] 더 많은 API 엔드포인트
- [ ] WebSocket 지원
- [ ] 데이터베이스 연동
- [ ] 모니터링 대시보드

## 📄 라이선스

MIT License

## 🤝 기여

Issues와 Pull Requests를 환영합니다!