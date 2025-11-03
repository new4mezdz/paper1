# -*- coding: utf-8 -*-
"""
fountain/auto_blocksize.py
自动选择 block_size，并添加“容量护栏”（cap guardrail）：
- 支持 cap_bytes_override（如来自视频首帧探测）
- 对 auto / manual / force_k 场景统一夹紧到 max_bs = capB - overhead
- 当超限被夹紧或自动调大 k 时，给出可读的告警与反推建议
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Union

import numpy as np
from PIL import Image

# ---------------- dataclass ----------------

@dataclass
class AutoBSResult:
    chosen_block_size: int
    frame_capacity_bytes: int
    max_block_size: int
    k_estimate: int
    reason: str            # "auto" / "manual" / "force_k" / "force_k_clipped"
    clipped: bool          # 是否因容量而被夹紧/调整
    advice: str            # 反推建议（最小分辨率/槽位等）
    cap_source: str        # "override" / "cover" / "unknown"

# ---------------- capacity helpers ----------------

def compute_frame_capacity_bytes(
    cover_img: Image.Image,
    bits_per_pixel: int = 1,
    headroom: float = 0.98
) -> int:
    """LSB 模拟：像素数 * 每像素可用 bit 数 /8 * headroom。"""
    arr = np.array(cover_img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    cap_bits = h * w * max(1, int(bits_per_pixel))
    cap_bytes = int((cap_bits // 8) * float(headroom))
    return max(0, cap_bytes)

def _advice_for_exceed(block_size: int, capB: int, blocks: Optional[int]=None,
                       slots_per_block: Optional[int]=None) -> str:
    """给出反推建议字符串（尽量通用，不假设 DCT 或 LSB）。"""
    if capB <= 0:
        return "未获取到单帧容量；建议提供 cap_bytes_override 或封面图用于估算。"
    needB = block_size + 26
    if needB <= capB:
        return ""
    # 能写下该 block_size 的最低容量需求
    need_capB = needB
    tips = [f"单帧容量不足：需要≥{need_capB}B，当前≈{capB}B。"]
    if blocks and slots_per_block:
        # DCT 模式粗略反推：所需每块槽位数 or 所需分辨率（块数）
        import math
        need_bits = needB * 8
        spb_min = math.ceil(need_bits / max(1, blocks))
        tips.append(f"如为 DCT：在当前分辨率下需每块槽位≥{spb_min}。")
        blk_min = math.ceil(need_bits / max(1, slots_per_block))
        # 分辨率估算为 8x8 块个数 → 近似成 sqrt
        tips.append(f"或需总 8×8 块数≥{blk_min}（例如把分辨率提升到约 √({blk_min}×64) 像素级别）。")
    return " ".join(tips)

# ---------------- main logic ----------------

def parse_block_size_opt(opt: Optional[Union[str, int]]) -> Optional[Union[str, int]]:
    if opt is None:
        return None
    if isinstance(opt, int):
        return int(opt)
    s = str(opt).strip().lower()
    if s in ("auto", "0"):
        return "auto"
    if s.isdigit():
        return int(s)
    raise ValueError(f"无法解析 block_size 选项: {opt!r}")

def auto_block_size(
    msg_len: int,
    capB: int,
    target_k: int = 200,
    min_bs: int = 128,
    align: int = 16,
) -> Tuple[int, int]:
    """基于给定 capB 进行 auto：返回 (bs, k_est)（未裁到 msg_len）。"""
    import math
    max_bs = max(1, capB - 26)
    ideal_bs = max(1, math.ceil(int(msg_len) / max(1, int(target_k))))
    bs = max(int(min_bs), min(int(max_bs), int(ideal_bs)))
    if align and align > 1:
        bs = (bs + align - 1) // align * align
    k_est = max(1, math.ceil(int(msg_len) / bs))
    return int(bs), int(k_est)

def resolve_block_size(
    arg_block_size: Optional[Union[str, int]],
    msg_len: int,
    cover_img: Optional[Image.Image],
    target_k: int = 200,
    headroom: float = 0.98,
    min_bs: int = 128,
    align: int = 16,
    overhead: int = 26,
    bits_per_pixel: int = 1,
    force_k: Optional[int] = None,
    cap_bytes_override: Optional[int] = None,
    # 为了生成更具体的建议（可选，来自视频探测）
    dct_blocks: Optional[int] = None,          # ⌊H/8⌋*⌊W/8⌋
    dct_slots_per_block: Optional[int] = None  # 你配置的每块槽位
) -> AutoBSResult:
    """
    统一处理 block_size + 容量护栏：
    - 若提供 cap_bytes_override，则优先用它作为 capB；否则尝试 cover_img（LSB估算）；再否则 capB=0。
    - force_k：若 bs=ceil(msg_len/force_k) > max_bs，则自动提升到 k_eff=ceil(msg_len/max_bs) 并告警。
    - manual：若 bs>max_bs，直接夹紧并告警。
    - auto：按 capB 计算 ideal_bs 并落在 [min_bs, max_bs]，再裁到 ≤ msg_len。
    """
    # 1) 拿 capB
    cap_src = "unknown"
    if cap_bytes_override is not None:
        capB = int(cap_bytes_override); cap_src = "override"
    elif cover_img is not None:
        capB = compute_frame_capacity_bytes(cover_img, bits_per_pixel=bits_per_pixel, headroom=headroom)
        cap_src = "cover"
    else:
        capB = 0; cap_src = "unknown"

    max_bs = max(1, int(capB) - int(overhead))

    # 2) force_k 模式
    if force_k is not None and int(force_k) > 0:
        import math
        fk = int(force_k)
        bs0 = max(1, math.ceil(int(msg_len) / fk))
        clipped = False
        advice = ""
        if capB > 0 and bs0 > max_bs:
            # 自动提高 k 以适配容量
            k_eff = max(1, math.ceil(int(msg_len) / max_bs))
            bs = max(1, math.ceil(int(msg_len) / k_eff))
            clipped = True
            advice = _advice_for_exceed(bs0, capB, dct_blocks, dct_slots_per_block)
            return AutoBSResult(
                chosen_block_size=int(bs),
                frame_capacity_bytes=int(capB),
                max_block_size=int(max_bs),
                k_estimate=int(k_eff),
                reason="force_k_clipped",
                clipped=clipped,
                advice=advice,
                cap_source=cap_src,
            )
        # 否则按原 force_k
        bs = bs0
        k_est = max(1, math.ceil(int(msg_len) / bs))
        return AutoBSResult(int(bs), int(capB), int(max_bs), int(k_est),
                            reason="force_k", clipped=False, advice="", cap_source=cap_src)

    # 3) 解析 block_size 其他模式
    parsed = parse_block_size_opt(arg_block_size)

    # 3a) auto
    if parsed == "auto" or parsed is None:
        bs, k_est = auto_block_size(
            msg_len=msg_len,
            capB=capB if capB>0 else (msg_len + overhead),  # cap 不详时退化给足
            target_k=target_k, min_bs=min_bs, align=align
        )
        bs = min(bs, max(1, int(msg_len)))  # 小消息允许 k=1
        return AutoBSResult(int(bs), int(capB), int(max_bs), int(k_est),
                            reason="auto", clipped=False, advice="", cap_source=cap_src)

    # 3b) manual
    bs = int(parsed)
    clipped = False
    advice = ""
    if capB > 0 and bs > max_bs:
        bs = max_bs
        clipped = True
        advice = _advice_for_exceed(int(parsed), capB, dct_blocks, dct_slots_per_block)
    if align and align > 1:
        bs = (bs + align - 1) // align * align
    bs = min(bs, max(1, int(msg_len)))
    import math
    k_est = max(1, math.ceil(int(msg_len) / bs))
    return AutoBSResult(int(bs), int(capB), int(max_bs), int(k_est),
                        reason="manual", clipped=clipped, advice=advice, cap_source=cap_src)
