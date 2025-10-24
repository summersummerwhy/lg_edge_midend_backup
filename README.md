# Topst D3

## 1.필수 패키지 설치
sudo apt update
sudo apt install -y python3 python3-venv python3-pip mosquitto mosquitto-clients

## 2. Mosquitto 설정
(1) 프로젝트의 설정 파일 복사

프로젝트 폴더 내의 mosquitto/ 디렉토리를 시스템 경로로 복사합니다.

sudo cp -r ./mosquitto /etc/mosquitto/
sudo chown -R mosquitto:mosquitto /etc/mosquitto

(2) Mosquitto 서비스 재시작
sudo systemctl restart mosquitto

(3) 서비스 상태 확인
sudo systemctl status mosquitto


정상적으로 실행 중이라면 다음과 같은 로그가 표시됩니다:

Active: active (running)

## 3. Python 환경 설정
(1) 가상환경 생성 및 활성화
cd ~/topst   # 프로젝트 루트 경로로 이동
python3 -m venv venv
source venv/bin/activate

(2) 의존성 설치
pip install --upgrade pip
pip install fastapi uvicorn asyncio-mqtt pydantic


또는 requirements.txt가 있다면:

pip install -r requirements.txt

## 4. 환경 변수 설정

TOPST 서버가 MQTT 브로커에 접근할 수 있도록 환경변수를 설정합니다.

export MQTT_HOST=localhost
export MQTT_PORT=1883
# export MQTT_USER=server
# export MQTT_PASS=server_pass

## 5. 서버 실행

FastAPI 개발 서버를 실행합니다.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

실행 확인

API 문서: http://localhost:8000/docs

MQTT 브로커 연결 테스트:

mosquitto_sub -h localhost -t "#" -v
mosquitto_pub -h localhost -t "test/topic" -m "Hello, TOPST!"

## 6. 서버 종료
deactivate     # Python 가상환경 종료
sudo systemctl stop mosquitto

## 프로젝트 구조 예시
topst/
 ├─ app/
 │   ├─ main.py
 │   └─ ...
 ├─ data/
 ├─ mosquitto/
 │   ├─ mosquitto.conf
 │   ├─ passwd
 │   └─ acl
 ├─ requirements.txt
 └─ README.md

## 요약 명령어
sudo apt install -y mosquitto mosquitto-clients python3 python3-venv python3-pip
sudo cp -r ./mosquitto /etc/mosquitto/
sudo chown -R mosquitto:mosquitto /etc/mosquitto
sudo systemctl restart mosquitto
cd topst
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export MQTT_HOST=localhost
export MQTT_PORT=1883
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload