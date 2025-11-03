# -*- coding: utf-8 -*-
"""
单帧 DCT-QIM 隐写（纯 PRNG 选位版）
- 载体：8x8 DCT 的一个 AC 系数（默认 (3,3)）。
- 选位与图像内容无关，仅由 (H, W, seed, 需要块数) 决定，嵌/取位置严格一致。
- 每比特重复 repeat 次并多数投票。
"""
from __future__ import annotations

import random
from typing import List, Tuple

import cv2
import numpy as np


# ----------------- 基础 DCT/IDCT -----------------
def _block_view(arr: np.ndarray, block=(8, 8)) -> np.ndarray:
    h, w = arr.shape
    bh, bw = block
    assert h % bh == 0 and w % bw == 0
    return arr.reshape(h // bh, bh, w // bw, bw).swapaxes(1, 2)  # (nbh, nbw, bh, bw)


def _merge_blocks(blocks: np.ndarray) -> np.ndarray:
    nbh, nbw, bh, bw = blocks.shape
    return blocks.swapaxes(1, 2).reshape(nbh * bh, nbw * bw)


def _dct2(block: np.ndarray) -> np.ndarray:
    return cv2.dct(block.astype(np.float32))


def _idct2(block: np.ndarray) -> np.ndarray:
    return cv2.idct(block.astype(np.float32))


# ----------------- 位置选取（与内容无关，仅由 seed 决定） -----------------
def _select_blocks_uniform(H: int, W: int, needed: int, seed: int) -> List[Tuple[int, int]]:
    """
    在所有 8x8 块里等概率抽样 needed 个不同块（顺序由 seed 决定）。
    """
    assert H % 8 == 0 and W % 8 == 0, "image size must be divisible by 8"
    nbh, nbw = H // 8, W // 8
    nblocks = nbh * nbw
    assert 0 < needed <= nblocks, f"needed={needed} exceeds capacity={nblocks}"
    rng = random.Random(seed)
    idxs = rng.sample(range(nblocks), needed)  # 唯一且打乱（顺序稳定）
    return [(idx // nbw, idx % nbw) for idx in idxs]


# ----------------- QIM 嵌入/提取 -----------------
def _qim_embed(coeff: float, bit: int, delta: float) -> float:
    # 奇偶量化：使 round(c/Δ) 的奇偶性 == bit
    q = int(round(coeff / delta))
    if (q & 1) != int(bit):
        q = q + 1 if coeff >= 0 else q - 1
    return float(q) * float(delta)


def _qim_extract(coeff: float, delta: float) -> int:
    q = int(round(coeff / delta))
    return q & 1


def embed_bits_to_image(
    img_bgr: np.ndarray,
    bits: np.ndarray,
    repeat: int = 3,
    delta: float = 6.0,
    carrier_pos: Tuple[int, int] = (3, 3),
    seed: int = 1234,
) -> np.ndarray:
    """
    在图像 BGR 上嵌入 bits（长度 L），每比特重复 repeat 次并做投票。
    约束：needed = L * repeat <= nblocks = (H/8)*(W/8)
    """
    H, W, _ = img_bgr.shape
    assert H % 8 == 0 and W % 8 == 0, "image size must be divisible by 8"

    # 仅修改亮度 Y（色度不改）
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0].astype(np.float32)

    L = int(len(bits))
    needed = L * int(repeat)
    pos_blocks = _select_blocks_uniform(H, W, needed, seed)

    blocks = _block_view(Y, (8, 8)).copy()
    bi, bj = carrier_pos

    idx = 0
    for b in bits:
        bb = int(b)
        for _ in range(repeat):
            i, j = pos_blocks[idx]
            D = _dct2(blocks[i, j])
            D[bi, bj] = _qim_embed(float(D[bi, bj]), bb, float(delta))
            blocks[i, j] = _idct2(D)
            idx += 1

    Y2 = _merge_blocks(blocks)
    ycc[:, :, 0] = np.clip(Y2, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    return out


def extract_bits_from_image(
    img_bgr: np.ndarray,
    num_bits: int,
    repeat: int = 3,
    delta: float = 6.0,
    carrier_pos: Tuple[int, int] = (3, 3),
    seed: int = 1234,
) -> np.ndarray:
    """
    从图像提取 num_bits（必须与嵌入端的 bits_per_frame 一致）；位置与嵌入端一致。
    """
    H, W, _ = img_bgr.shape
    assert H % 8 == 0 and W % 8 == 0, "image size must be divisible by 8"

    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0].astype(np.float32)

    needed = int(num_bits) * int(repeat)
    pos_blocks = _select_blocks_uniform(H, W, needed, seed)
    blocks = _block_view(Y, (8, 8))
    bi, bj = carrier_pos

    votes = np.zeros((num_bits, 2), dtype=int)  # [count0, count1]
    idx = 0
    for t in range(num_bits):
        for _ in range(repeat):
            i, j = pos_blocks[idx]
            D = _dct2(blocks[i, j])
            bit = _qim_extract(float(D[bi, bj]), float(delta))
            votes[t, bit] += 1
            idx += 1

    out_bits = (votes[:, 1] >= votes[:, 0]).astype(np.uint8)
    return out_bits


# 便利的 IO 包装
def embed_bits_file(in_path: str, out_path: str, bits: np.ndarray, repeat=3, delta=6.0, seed=1234):
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    assert img is not None, f"cannot read image: {in_path}"
    out = embed_bits_to_image(img, bits, repeat=repeat, delta=delta, seed=seed)
    ok = cv2.imwrite(out_path, out)
    assert ok, f"cannot write image: {out_path}"


def extract_bits_file(in_path: str, num_bits: int, repeat=3, delta=6.0, seed=1234) -> np.ndarray:
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    assert img is not None, f"cannot read image: {in_path}"
    return extract_bits_from_image(img, num_bits=num_bits, repeat=repeat, delta=delta, seed=seed)
