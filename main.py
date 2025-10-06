from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import paho.mqtt.client as mqtt
import json
import logging
from datetime import datetime
from collections import deque

# 로깅
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
mqtt_client = None
recent_events = deque(maxlen=50)
device_status = {
    "esp32": {"connected": False, "last_seen": None},
    "topst": {"connected": False, "last_seen": None}
}

# MQTT 설정
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# MQTT 콜백
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("✅ MQTT 연결 성공")
        client.subscribe("lowend/#")
    else:
        logger.error(f"❌ MQTT 연결 실패: {rc}")

def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        data = json.loads(msg.payload.decode())
        
        # ESP32 상태 업데이트
        device_status["esp32"]["connected"] = True
        device_status["esp32"]["last_seen"] = datetime.now().isoformat()
        
        logger.info(f"📩 수신: {topic}")
        
        # 토픽별 처리
        if "motion" in topic and data.get("detected"):
            handle_motion(client)
        elif "sound" in topic and data.get("level", 0) > 70:
            handle_sound(client, data.get("level"))
            
    except Exception as e:
        logger.error(f"❌ 오류: {e}")

def handle_motion(client):
    """모션 감지 처리"""
    logger.warning("🚨 모션 감지!")
    log_event("motion", "모션 감지")
    send_alert(client, "motion", "medium")

def handle_sound(client, level):
    """소리 감지 처리"""
    logger.warning(f"🔊 소리 감지: {level}dB")
    log_event("sound", f"소리 감지 ({level}dB)")
    send_alert(client, "sound", "low")

def send_alert(client, alert_type, severity):
    """High-end로 알림 전송"""
    alert = {
        "alert_type": alert_type,
        "severity": severity,
        "timestamp": datetime.now().isoformat()
    }
    client.publish("highend/alert", json.dumps(alert), qos=1)
    logger.info(f"📤 알림 전송: {severity}")
    log_event("alert", f"{severity} 알림 전송")

def log_event(event_type, message):
    """이벤트 로그"""
    recent_events.append({
        "type": event_type,
        "message": message,
        "timestamp": datetime.now().isoformat()
    })

# FastAPI 시작/종료
@app.on_event("startup")
async def startup():
    global mqtt_client
    mqtt_client = mqtt.Client("rpi4_security")
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
    logger.info("🚀 서버 시작")

@app.on_event("shutdown")
async def shutdown():
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    logger.info("🛑 서버 종료")

# API 엔드포인트
@app.get("/")
async def dashboard():
    """웹 대시보드"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>방범 카메라</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial; margin: 20px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }
            .card { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; }
            .status-ok { color: #27ae60; }
            .status-error { color: #e74c3c; }
            .event { padding: 10px; border-bottom: 1px solid #eee; }
            button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏠 방범 카메라 모니터링</h1>
            </div>
            
            <div class="card">
                <h2>시스템 상태</h2>
                <p>ESP32: <span id="esp32-status">확인 중...</span></p>
                <p>TOPST: <span id="topst-status">확인 중...</span></p>
                <p>MQTT: <span id="mqtt-status">확인 중...</span></p>
            </div>
            
            <div class="card">
                <h2>최근 이벤트 <button onclick="loadEvents()">새로고침</button></h2>
                <div id="events">로딩 중...</div>
            </div>
        </div>
        
        <script>
            async function loadStatus() {
                const res = await fetch('/api/status');
                const data = await res.json();
                
                document.getElementById('esp32-status').innerHTML = 
                    data.esp32.connected ? '<span class="status-ok">✅ 연결됨</span>' : '<span class="status-error">❌ 연결 끊김</span>';
                document.getElementById('topst-status').innerHTML = 
                    data.topst.connected ? '<span class="status-ok">✅ 연결됨</span>' : '<span class="status-error">❌ 연결 끊김</span>';
                document.getElementById('mqtt-status').innerHTML = 
                    data.mqtt ? '<span class="status-ok">✅ 정상</span>' : '<span class="status-error">❌ 오류</span>';
            }
            
            async function loadEvents() {
                const res = await fetch('/api/events');
                const events = await res.json();
                
                const html = events.map(e => `
                    <div class="event">
                        <strong>${e.message}</strong>
                        <br><small>${new Date(e.timestamp).toLocaleString('ko-KR')}</small>
                    </div>
                `).join('');
                
                document.getElementById('events').innerHTML = html || '<p>이벤트 없음</p>';
            }
            
            setInterval(() => {
                loadStatus();
                loadEvents();
            }, 5000);
            
            loadStatus();
            loadEvents();
        </script>
    </body>
    </html>
    """)

@app.get("/api/status")
async def get_status():
    """시스템 상태"""
    return {
        "esp32": device_status["esp32"],
        "topst": device_status["topst"],
        "mqtt": mqtt_client.is_connected() if mqtt_client else False
    }

@app.get("/api/events")
async def get_events():
    """최근 이벤트"""
    return list(recent_events)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)