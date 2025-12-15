import contextlib
import asyncio

from fastapi import FastAPI

from .routes import router
from .mqtt.worker import mqtt_worker
from .mqtt.publisher import disconnect_client
from .logging_setup import setup_logging
from .config import LOG_LEVEL

setup_logging(LOG_LEVEL)

app = FastAPI(title="TOPST Receiver (asyncio-mqtt)", version="0.3.0")
app.include_router(router)


@app.on_event("startup")
async def on_startup():
    app.state.mqtt_task = asyncio.create_task(mqtt_worker())
    # 로거가 있으므로 print 대신 로그 사용
    import logging

    logging.getLogger(__name__).info("[STARTUP] MQTT worker launched.")
    logging.getLogger("app.mqtt.publisher").setLevel(logging.WARNING)
    logging.getLogger("app.handlers").setLevel(logging.WARNING)


@app.on_event("shutdown")
async def on_shutdown():
    task: asyncio.Task = getattr(app.state, "mqtt_task", None)
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    import logging

    logging.getLogger(__name__).info("[SHUTDOWN] MQTT worker stopped.")
    await disconnect_client()
