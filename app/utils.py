from datetime import datetime
from pathlib import Path


def ts_to_iso(ts_ms: int) -> str:
    return datetime.utcfromtimestamp(ts_ms / 1000).isoformat() + "Z"


def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def save_wav_pcm16_mono(pcm_bytes: bytes, sample_rate: int, out_path: Path) -> None:
    import wave

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
