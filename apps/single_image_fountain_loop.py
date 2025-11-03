# -*- coding: utf-8 -*-
"""
单图循环：每轮把一个 LT 包嵌入到封面图像，立刻提取并投入译码；
当累计包数足以复原原文时停止；每轮输出一张 iter_XXXX.png。

更新点（方案B）：
- 读取封面后，若尺寸不是 8 的倍数，自动用 BORDER_REPLICATE 补到最近的 8 倍数，并打印提示。
- 其余逻辑保持不变。
"""
from __future__ import annotations

# —— 关键：无条件把项目根加入 sys.path（避免相对运行时找不到包）
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from fountain.lt_min import LTEncoder, LTDecoder
from glue.fountain_glue import (
    pack_packet_to_bytes, unpack_packet_from_bytes,
    bytes_to_bits, bits_to_bytes, frame_pad_bits,
)
from stego.stego_qim_frame import embed_bits_to_image, extract_bits_from_image


def capacity_bits(img_shape, repeat: int) -> int:
    """在 1 比特/块、重复 repeat 次时的最大可嵌入比特数（要求已是 8 的倍数）。"""
    H, W, _ = img_shape
    assert H % 8 == 0 and W % 8 == 0, "cover size must be divisible by 8"
    nblocks = (H // 8) * (W // 8)
    return nblocks // max(1, int(repeat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cover', required=True, help='封面图像（任意尺寸，脚本会自动补到 8 的倍数）')
    ap.add_argument('--message', required=True, help='要嵌入/恢复的消息文件')
    ap.add_argument('--outdir', default='data/stego_rounds')
    ap.add_argument('--bits-per-frame', type=int, default=4096, help='<=0 自动容量')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--delta', type=float, default=6.0)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--block-size', type=int, default=512, help='LT 源块字节数')
    ap.add_argument('--max-rounds', type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 读消息
    with open(args.message, 'rb') as f:
        msg = f.read()
    enc = LTEncoder(msg, block_size=args.block_size, base_seed=args.seed, systematic=True)
    dec = LTDecoder()

    # 读封面并自动补到 8 的倍数
    cover = cv2.imread(args.cover, cv2.IMREAD_COLOR)
    assert cover is not None, f"cannot read cover: {args.cover}"
    H, W, _ = cover.shape
    if H % 8 != 0 or W % 8 != 0:
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        cover = cv2.copyMakeBorder(cover, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        H, W, _ = cover.shape
        print(f"[INFO] cover padded to {H}x{W} (multiple of 8)")

    # 计算容量并矫正 bits-per-frame
    cap = capacity_bits(cover.shape, args.repeat)
    frame_bits = cap if args.bits_per_frame <= 0 else min(args.bits_per_frame, cap)
    if args.bits_per_frame > cap:
        print(f"[WARN] bits-per-frame={args.bits_per_frame} 超过容量 {cap}，已降为 {frame_bits}。")
    if frame_bits <= 0:
        raise ValueError("frame_bits<=0，图片尺寸或 repeat 设置不合理。")

    for r in tqdm(range(args.max_rounds), desc='embed/recv'):
        # 取下一个喷泉包并封装为位流（定长帧）
        pkt = enc.next_packet()
        pkt_bytes = pack_packet_to_bytes(pkt)
        bits = bytes_to_bits(pkt_bytes)
        bits = frame_pad_bits(bits, frame_bits)

        # 嵌入并输出当前“帧”
        out_img = embed_bits_to_image(cover, bits, repeat=args.repeat, delta=args.delta, seed=args.seed)
        out_path = os.path.join(args.outdir, f"iter_{r:04d}.png")
        ok = cv2.imwrite(out_path, out_img)
        assert ok, f"write fail: {out_path}"

        # 立刻模拟接收端提取本帧并尝试 LT 剥皮译码
        recv_bits = extract_bits_from_image(out_img, num_bits=frame_bits, repeat=args.repeat, delta=args.delta, seed=args.seed)
        recv_bytes = bits_to_bytes(recv_bits)
        try:
            rpkt = unpack_packet_from_bytes(recv_bytes)
        except Exception:
            # 可能由于提取误码导致头解析失败，跳过此帧
            continue
        dec.add_packet(rpkt)
        if dec.is_decoded():
            rec = dec.reconstruct()
            out_bin = os.path.join(args.outdir, 'recovered.bin')
            with open(out_bin, 'wb') as f:
                f.write(rec)
            print(f"\n[OK] Decoded after {r+1} frames -> {out_bin}")
            return

    print("[WARN] 未能在 max-rounds 内解出，请增加轮数或调整 Δ/repeat/容量。")


if __name__ == '__main__':
    main()
