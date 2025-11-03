from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    frames: Optional[int]
    ok: bool
    backend: str

def probe_video_meta(path: str) -> VideoMeta:
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
            if w <= 0 or h <= 0:
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
            cap.release()
            return VideoMeta(w,h,fps,frames, ok=(w>0 and h>0), backend="opencv")
    except Exception:
        pass
    try:
        import imageio.v2 as iio
        rdr = iio.get_reader(path)
        meta = rdr.get_meta_data()
        w = int(meta.get('size', (0,0))[0] or 0)
        h = int(meta.get('size', (0,0))[1] or 0)
        fps = float(meta.get('fps') or 0.0)
        frames = int(meta.get('nframes') or 0) or None
        if w <= 0 or h <= 0:
            im = rdr.get_data(0)
            h, w = im.shape[:2]
        rdr.close()
        return VideoMeta(w,h,fps,frames, ok=(w>0 and h>0), backend="imageio")
    except Exception:
        pass
    return VideoMeta(0,0,0.0,None, ok=False, backend="none")

def estimate_frame_capacity_bytes(
    H: int, W: int,
    mode: str = "dct",
    *,
    slots_per_block: int = 8,
    bits_per_pixel: int = 1,
    headroom: float = 0.98
) -> int:
    if H <= 0 or W <= 0: return 0
    mode = (mode or "dct").lower()
    if mode == "lsb":
        cap_bits = H * W * max(1, int(bits_per_pixel))
    else:
        blocks = (H // 8) * (W // 8)
        cap_bits = blocks * max(1, int(slots_per_block))
    return int((cap_bits // 8) * float(headroom))
