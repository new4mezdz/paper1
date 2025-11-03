# -*- coding: utf-8 -*-
# tools/qim_ber_probe.py
"""
测单帧 QIM 的比特错率（BER）。
"""
from __future__ import annotations

# —— 关键：无条件把项目根加入 sys.path（避免相对运行时找不到包）
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import cv2
import numpy as np

from stego.stego_qim_frame import embed_bits_to_image, extract_bits_from_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", required=True)
    ap.add_argument("--bits", type=int, default=256)   # 测 N 个比特
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--delta", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--carrier-row", type=int, default=3)
    ap.add_argument("--carrier-col", type=int, default=3)
    args = ap.parse_args()

    img = cv2.imread(args.cover, cv2.IMREAD_COLOR)
    assert img is not None, f"cannot read cover: {args.cover}"
    H, W = img.shape[:2]
    # 自动补到 8 的倍数，保证与主脚本一致
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)

    bits = np.random.randint(0, 2, size=(args.bits,), dtype=np.uint8)

    stego = embed_bits_to_image(
        img, bits,
        repeat=args.repeat, delta=args.delta,
        carrier_pos=(args.carrier_row, args.carrier_col), seed=args.seed
    )
    rx = extract_bits_from_image(
        stego, num_bits=args.bits,
        repeat=args.repeat, delta=args.delta,
        carrier_pos=(args.carrier_row, args.carrier_col), seed=args.seed
    )

    err = int(np.count_nonzero(rx != bits))
    ber = err / float(args.bits)
    print(f"BER={ber:.6f} ({err}/{args.bits}), repeat={args.repeat}, delta={args.delta}, carrier=({args.carrier_row},{args.carrier_col})")


if __name__ == "__main__":
    main()
