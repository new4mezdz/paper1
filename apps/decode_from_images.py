# -*- coding: utf-8 -*-
"""
对 iter_*.png 目录批量提取与 LT 译码，直至恢复或遍历完。
"""
from __future__ import annotations

# —— 关键：无条件把项目根加入 sys.path
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import glob
import os
import cv2
from tqdm import tqdm

from fountain.lt_min import LTDecoder
from glue.fountain_glue import unpack_packet_from_bytes, bits_to_bytes
from stego.stego_qim_frame import extract_bits_from_image

def capacity_bits(img_shape, repeat: int) -> int:
    H, W, _ = img_shape
    if H % 8 != 0 or W % 8 != 0:
        raise ValueError('image size must be divisible by 8')
    nblocks = (H // 8) * (W // 8)
    return nblocks // max(1, int(repeat))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', required=True)
    ap.add_argument('--bits-per-frame', type=int, default=4096, help='<=0 自动容量')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--delta', type=float, default=6.0)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--out', default='decoded.bin')
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.indir, '*.png')))
    assert paths, f"no png found in {args.indir}"
    dec = LTDecoder()

    auto = args.bits_per_frame <= 0

    for p in tqdm(paths, desc='decode frames'):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        try:
            cap = capacity_bits(img.shape, args.repeat)
        except Exception:
            continue
        frame_bits = cap if auto else min(args.bits_per_frame, cap)
        if not auto and args.bits_per_frame > cap:
            print(f"[WARN] {os.path.basename(p)}: bits-per-frame={args.bits_per_frame} 超过容量 {cap}，已降为 {frame_bits}。")

        bits = extract_bits_from_image(img, num_bits=frame_bits, repeat=args.repeat, delta=args.delta, seed=args.seed)
        by = bits_to_bytes(bits)
        try:
            pkt = unpack_packet_from_bytes(by)
        except Exception:
            continue
        dec.add_packet(pkt)
        if dec.is_decoded():
            data = dec.reconstruct()
            with open(args.out, 'wb') as f:
                f.write(data)
            print(f"[OK] Recovered -> {args.out}")
            return
    print("[WARN] 没有足够的有效包。可尝试：提高 bits-per-frame、增大 delta、提高 repeat 或提供更多帧。")

if __name__ == '__main__':
    main()
