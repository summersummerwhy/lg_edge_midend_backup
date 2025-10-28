from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field, PositiveInt


class MotionPayload(BaseModel):
    motion: int = Field(ge=0, le=1)


class AudioPayload(BaseModel):
    format: str = Field(pattern="^(?i)PCM$")
    sample_rate: PositiveInt = 16000
    channel: PositiveInt = 1
    dur_ms: PositiveInt = 20
    data_b64: str


class CameraPayload(BaseModel):
    format: str = Field(pattern="^(?i)jpeg$")
    width: PositiveInt
    height: PositiveInt
    data_b64: Optional[str] = None
    part: Optional[Literal["start", "chunk", "end"]] = None
    idx: Optional[int] = None
    total: Optional[int] = None


class Envelope(BaseModel):
    device: str
    ts: int  # UTC ms
    seq: int
    payload: Dict[str, Any]
