#!/bin/bash

echo "=========================================="
echo "Raspberry Pi 4 서버 초기 설정 시작"
echo "=========================================="

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 시스템 업데이트
echo -e "${YELLOW}시스템 업데이트 중...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# 필요한 시스템 패키지 설치
echo -e "${YELLOW}시스템 패키지 설치 중...${NC}"
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    mosquitto \
    mosquitto-clients \
    build-essential \
    libopenblas-dev

# Mosquitto MQTT 브로커 시작 및 활성화
echo -e "${YELLOW}MQTT 브로커 설정 중...${NC}"
sudo systemctl enable mosquitto
sudo systemctl start mosquitto

# 가상환경 생성
echo -e "${YELLOW}Python 가상환경 생성 중...${NC}"
python3 -m venv venv
source venv/bin/activate

# Python 패키지 설치
echo -e "${YELLOW}Python 패키지 설치 중...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# 디렉토리 생성
echo -e "${YELLOW}필요한 디렉토리 생성 중...${NC}"
mkdir -p models
mkdir -p logs
mkdir -p data

# 환경변수 파일 생성 (없는 경우)
if [ ! -f .env ]; then
    echo -e "${YELLOW}.env 파일 생성 중...${NC}"
    cat > .env << EOF
# 서버 설정
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# MQTT 설정
MQTT_BROKER=localhost
MQTT_PORT=1883
MQTT_USERNAME=
MQTT_PASSWORD=

# 로깅
LOG_LEVEL=INFO
EOF
fi

# systemd 서비스 파일 생성
echo -e "${YELLOW}systemd 서비스 생성 중...${NC}"
sudo tee /etc/systemd/system/rpi-manager.service > /dev/null << EOF
[Unit]
Description=Raspberry Pi Manager Server
After=network.target mosquitto.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable rpi-manager.service

echo -e "${GREEN}=========================================="
echo -e "설정 완료!"
echo -e "==========================================${NC}"
echo ""
echo "사용 가능한 명령어:"
echo "  - 서버 시작: sudo systemctl start rpi-manager"
echo "  - 서버 중지: sudo systemctl stop rpi-manager"
echo "  - 서버 상태: sudo systemctl status rpi-manager"
echo "  - 로그 확인: sudo journalctl -u rpi-manager -f"
echo ""
echo "또는 수동 실행:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""