# -*- coding: utf-8 -*-
"""
层间胶水：
- LT 包 <-> 位流 的打包/解包；
- 定长帧（bits_per_frame）填充；
"""
from __future__ import annotations
from typing import Optional
import struct
import numpy as np

from fountain.old import LTPacket  # 关键：从包名导入

MAGIC = 0xA5  # 8 bits
VERSION = 0x1  # 4 bits
FLAG_SYS = 0x1  # bit0 表示系统包

# Header: magic(8) | ver(4) | flags(4) | k(16) | block_size(16) | msg_len(32)
# Sys=0 附加 seed(32)；Sys=1 附加 sys_idx(16)

def pack_packet_to_bytes(pkt: LTPacket) -> bytes:
    flags = 0
    if pkt.sys_idx is not None:
        flags |= FLAG_SYS
        body = struct.pack(">H", pkt.sys_idx)
    else:
        body = struct.pack(">I", pkt.seed & 0xFFFFFFFF)  # type: ignore
    head = struct.pack(
        ">B B H H I",
        MAGIC,
        ((VERSION & 0xF) << 4) | (flags & 0xF),
        pkt.k & 0xFFFF,
        pkt.block_size & 0xFFFF,
        pkt.msg_len & 0xFFFFFFFF,
    )
    return head + body + pkt.payload

def unpack_packet_from_bytes(buf: bytes) -> LTPacket:
    if len(buf) < 1 + 1 + 2 + 2 + 4:
        raise ValueError("buffer too small for header")
    magic = buf[0]
    if magic != MAGIC:
        raise ValueError("bad magic")
    ver_flags = buf[1]
    ver = (ver_flags >> 4) & 0xF
    flags = ver_flags & 0xF
    if ver != VERSION:
        raise ValueError("bad version")
    k, block_size, msg_len = struct.unpack(">H H I", buf[2:2+2+2+4])
    pos = 2 + 2 + 2 + 4
    sys_mode = (flags & FLAG_SYS) != 0
    if sys_mode:
        if len(buf) < pos + 2:
            raise ValueError("buffer too small for sys_idx")
        (sys_idx,) = struct.unpack(">H", buf[pos:pos+2])
        pos += 2
        seed = None
    else:
        if len(buf) < pos + 4:
            raise ValueError("buffer too small for seed")
        (seed,) = struct.unpack(">I", buf[pos:pos+4])
        pos += 4
        sys_idx = None
    if len(buf) < pos + block_size:
        raise ValueError("buffer too small for payload")
    payload = buf[pos:pos+block_size]
    return LTPacket(k=k, block_size=block_size, msg_len=msg_len, seed=seed, sys_idx=sys_idx, payload=payload)

# ------------ 位流/字节流转换 --------------
def bytes_to_bits(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits.astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = bits.astype(np.uint8)
    if len(bits) % 8 != 0:
        pad = 8 - (len(bits) % 8)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    by = np.packbits(bits)
    return by.tobytes()

def frame_pad_bits(bits: np.ndarray, bits_per_frame: int) -> np.ndarray:
    if len(bits) > bits_per_frame:
        return bits[:bits_per_frame]
    if len(bits) < bits_per_frame:
        pad = np.zeros(bits_per_frame - len(bits), dtype=np.uint8)
        return np.concatenate([bits, pad])
    return bits
