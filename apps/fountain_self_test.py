# -*- coding: utf-8 -*-
"""
喷泉码自检脚本（仅 Fountain 层，不经过隐写）
- 丢包信道（独立丢包率 p_loss）
- until-success：发到解出为止；fixed：发 ceil(k*(1+overhead)) 个
- 默认从 data/samples/msg.txt 读取消息；成功后打印解码结果
- 自动 block_size：根据消息长度选择 block_size，使 k 落在 [k_min,k_max] 内（尽量）
- 输出期望值：
  - until-success：E[T|R]=R/(1-p)，Std[T|R]=sqrt(R*p)/(1-p)
  - fixed：E[rx|T]=T*(1-p)
"""
from __future__ import annotations

import sys, os, math, random, time
import numpy as np

# 保证能找到项目包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fountain.lt_min import LTEncoder, LTDecoder

DEF_MSG_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'samples', 'msg.txt'))

def choose_block_size_auto(msg_len: int,
                           k_min: int = 8, k_max: int = 64,
                           allowed_sizes = (8,16,32,64,128,256,512,1024)) -> tuple[int,int]:
    """
    根据消息字节数自动选择 block_size（从 allowed_sizes 里挑），
    使 k=ceil(msg_len/block_size) 落在 [k_min,k_max] 内（尽量靠近中间）。
    返回 (block_size, k)
    """
    assert msg_len > 0
    cand = []
    for bs in allowed_sizes:
        if bs <= 0: continue
        k = max(1, (msg_len + bs - 1) // bs)
        cand.append((bs, k))
    mid = 0.5 * (k_min + k_max)
    in_range = [(bs, k) for (bs, k) in cand if k_min <= k <= k_max]
    if in_range:
        # 选 k 最接近中点的；若并列，选 block_size 更大的（负载更大，包更少）
        in_range.sort(key=lambda x: (abs(x[1]-mid), -x[0]))
        return in_range[0]
    # 不在范围：如果所有 k 都小于 k_min（消息很短），尽量把 k 拉高 —— 选最小的 block_size
    too_small = all(k < k_min for _, k in cand)
    if too_small:
        bs, k = min(cand, key=lambda x: x[0])  # 最小 block_size
        return bs, k
    # 否则 k 都太大（消息很长）：选最大的 block_size 把 k 尽量压低
    bs, k = max(cand, key=lambda x: x[0])
    return bs, k

def run_once(message: bytes,
             block_size: int,
             base_seed: int = 1234,
             loss: float = 0.2,
             mode: str = "until-success",
             overhead: float = 0.2,
             max_send_factor: float = 10.0,
             rng_seed: int = 20240829):
    assert 0.0 <= loss < 1.0
    mode = mode.lower().strip()
    assert mode in ("until-success", "fixed")

    enc = LTEncoder(message, block_size=block_size, base_seed=base_seed, systematic=True)
    dec = LTDecoder()
    k = enc.k
    rng = random.Random(rng_seed)

    if mode == "fixed":
        max_send = int(math.ceil(k * (1.0 + overhead)))
    else:
        max_send = int(math.ceil(k * max_send_factor))

    sent = recvd = 0
    t0 = time.time()
    while sent < max_send:
        pkt = enc.next_packet(); sent += 1
        if rng.random() < loss:
            continue  # 丢包
        recvd += 1
        dec.add_packet(pkt)
        if dec.is_decoded():
            rec = dec.reconstruct()
            ok = (rec == message)
            dt = time.time() - t0
            return {
                "ok": ok, "decoded": rec,
                "k": k, "block_size": block_size, "msg_len": len(message),
                "loss": loss, "mode": mode, "overhead": overhead if mode=="fixed" else None,
                "sent": sent, "received": recvd,
                "rx_over_k": recvd/float(k), "tx_over_k": sent/float(k),
                "time_sec": dt
            }

        if mode == "fixed" and sent >= max_send:
            break

    dt = time.time() - t0
    return {
        "ok": False, "decoded": b"",
        "k": k, "block_size": block_size, "msg_len": len(message),
        "loss": loss, "mode": mode, "overhead": overhead if mode=="fixed" else None,
        "sent": sent, "received": recvd,
        "rx_over_k": recvd/float(k) if k else 0.0, "tx_over_k": sent/float(k) if k else 0.0,
        "time_sec": dt
    }

def human_stats(s: dict) -> str:
    lines = []
    ok = s["ok"]; p = s["loss"]; q = 1.0 - p
    badge = "[OK]" if ok else "[FAIL]"
    lines.append(f"{badge} mode={s['mode']}  loss={p:.2f}  k={s['k']}  block_size={s['block_size']}  msg_len={s['msg_len']}")
    if s["mode"] == "fixed":
        lines.append(f"     overhead={s['overhead']:.2f}")
    lines.append(f"     sent={s['sent']}  received={s['received']}  rx/k={s['rx_over_k']:.3f}  tx/k={s['tx_over_k']:.3f}  time={s['time_sec']:.3f}s")

    # 期望值
    if s["mode"] == "until-success":
        R = max(1, s["received"])
        e_tx = R / q
        std_tx = math.sqrt(R * p) / q
        lines.append(f"     E[tx | R={R}] = {e_tx:.2f}  (±1σ ≈ {std_tx:.2f})")
        lines.append(f"     Δtx = actual - E ≈ {s['sent'] - e_tx:+.2f}")
    else:
        T = s["sent"]; e_rx = T * q
        lines.append(f"     E[rx | T={T}] = {e_rx:.2f}  (E[rx/k] ≈ {e_rx/float(s['k']):.3f})")

    # 解码结果（若成功）
    if ok:
        text = s["decoded"].decode("utf-8", errors="replace")
        hex32 = s["decoded"][:32].hex(" ")
        lines.append("----- DECODED MESSAGE (UTF-8) -----")
        lines.append(text)
        lines.append("----- HEX PREVIEW (first 32 bytes) -----")
        lines.append(hex32)

    return "\n".join(lines)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", help=f"消息文件（默认：{DEF_MSG_PATH}）")
    ap.add_argument("--block-size", type=int, default=0, help="LT 源块大小；0=自动选择")
    ap.add_argument("--k-min", type=int, default=8, help="自动选择目标 k 的下界")
    ap.add_argument("--k-max", type=int, default=64, help="自动选择目标 k 的上界")
    ap.add_argument("--allowed-sizes", default="8,16,32,64,128,256,512,1024", help="允许的 block_size 列表（字节），逗号分隔")
    ap.add_argument("--base-seed", type=int, default=1234)
    ap.add_argument("--loss", type=float, default=0.2)
    ap.add_argument("--mode", choices=["until-success","fixed"], default="until-success")
    ap.add_argument("--overhead", type=float, default=0.2)
    ap.add_argument("--max-send-factor", type=float, default=10.0)
    ap.add_argument("--rng-seed", type=int, default=20240829)
    args = ap.parse_args()

    # 准备消息
    msg_path = args.message or DEF_MSG_PATH
    if not os.path.isfile(msg_path):
        raise FileNotFoundError(f"找不到消息文件：{msg_path}")
    with open(msg_path, "rb") as f: message = f.read()

    # 自动/手动 block_size
    if args.block_size and args.block_size > 0:
        bs = int(args.block_size)
        k = max(1, (len(message) + bs - 1) // bs)
        print(f"[INFO] manual block_size={bs} -> k={k} (msg_len={len(message)})")
    else:
        allowed = tuple(int(x) for x in args.allowed_sizes.split(",") if x.strip())
        bs, k = choose_block_size_auto(len(message), args.k_min, args.k_max, allowed)
        print(f"[AUTO] chose block_size={bs} (allowed={allowed}) for msg_len={len(message)} -> k={k} (target {args.k_min}..{args.k_max})")

    stats = run_once(
        message=message, block_size=bs, base_seed=args.base_seed,
        loss=args.loss, mode=args.mode, overhead=args.overhead,
        max_send_factor=args.max_send_factor, rng_seed=args.rng_seed,
    )
    print(human_stats(stats))

if __name__ == "__main__":
    main()
