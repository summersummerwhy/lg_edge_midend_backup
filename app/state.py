import asyncio
from typing import Any, Dict, Tuple, Optional

# latest cache
latest: Dict[Tuple[str, str], Dict[str, Any]] = {}

# dedup (per kind)
last_seq: Dict[Tuple[str, str], int] = {}


def is_duplicate(
    device: str, kind: str, seq: int, *, idx: Optional[int] = None
) -> bool:
    """
    - 기본: (device, kind)의 마지막 seq보다 작거나 같으면 중복
    - 카메라는 청크 idx에 따라 호출측에서 별도 허용/제어
    """
    key = (device, kind)
    prev = last_seq.get(key)
    if prev is not None and seq <= prev:
        return True
    last_seq[key] = seq
    return False


# audio buffers
audio_buffer: Dict[str, list[bytes]] = {}
audio_last_ts: Dict[str, int] = {}
audio_start_ts: Dict[str, int] = {}

# MQTT publisher sequence + lock
seq_number = 0
mqtt_lock = asyncio.Lock()


# camera chunk store
class ChunkState:
    __slots__ = ("total", "received", "parts", "start_ts")

    def __init__(self, start_ts: int):
        self.total: Optional[int] = None
        self.received = set()
        self.parts: Dict[int, str] = {}  # idx -> b64
        self.start_ts = start_ts

    def set_total(self, total: int):
        if total is not None and total > 0:
            self.total = total

    def add(self, idx: int, data_b64: str):
        if idx not in self.received:
            self.received.add(idx)
            self.parts[idx] = data_b64

    def complete(self) -> bool:
        return self.total is not None and len(self.received) == self.total


camera_chunks: Dict[Tuple[str, int], ChunkState] = {}
