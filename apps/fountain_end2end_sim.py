# -*- coding: utf-8 -*-
"""
端到端喷泉仿真（经过 glue 的头部重复×N+CRC、字节↔比特、丢包+比特误码）
- 默认从 data/samples/msg.txt 读取消息；成功后打印解码结果
- 自动 block_size：根据消息长度选择 block_size，使 k 落在 [k_min,k_max] 内（尽量）
"""
from __future__ import annotations
import sys, os, math, random, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fountain.lt_min import LTEncoder, LTDecoder, LTPacket
from glue.fountain_glue import pack_packet_to_bytes, unpack_packet_from_bytes, bytes_to_bits, bits_to_bytes

DEF_MSG_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'samples', 'msg.txt'))

def choose_block_size_auto(msg_len: int,
                           k_min: int = 8, k_max: int = 64,
                           allowed_sizes = (8,16,32,64,128,256,512,1024)) -> tuple[int,int]:
    cand = []
    for bs in allowed_sizes:
        if bs <= 0: continue
        k = max(1, (msg_len + bs - 1) // bs)
        cand.append((bs, k))
    mid = 0.5 * (k_min + k_max)
    in_range = [(bs, k) for (bs, k) in cand if k_min <= k <= k_max]
    if in_range:
        in_range.sort(key=lambda x: (abs(x[1]-mid), -x[0]))
        return in_range[0]
    if all(k < k_min for _, k in cand):
        return min(cand, key=lambda x: x[0])
    return max(cand, key=lambda x: x[0])

def flip_bits(bits: np.ndarray, ber: float, rng: random.Random) -> np.ndarray:
    if ber <= 0.0: return bits
    flips = np.array([1 if rng.random() < ber else 0 for _ in range(len(bits))], dtype=np.uint8)
    return (bits ^ flips).astype(np.uint8)

def pkt_key(pkt: LTPacket):
    if pkt.sys_idx is not None: return ("sys", int(pkt.sys_idx))
    return ("seed", int(pkt.seed if pkt.seed is not None else -1))

def run_once(message: bytes, block_size: int, base_seed: int,
             loss: float, ber: float,
             mode: str, overhead: float, max_send_factor: float,
             rng_seed: int, allow_payload_errors: bool):
    enc = LTEncoder(message, block_size=block_size, base_seed=base_seed, systematic=True)
    dec = LTDecoder()
    k = enc.k
    rng = random.Random(rng_seed)

    truth_payload = {}
    if mode == "fixed":
        max_send = int(math.ceil(k * (1.0 + overhead)))
    else:
        max_send = int(math.ceil(k * max_send_factor))

    sent = recvd_valid = 0
    drop_loss = drop_hdr = drop_payload = 0
    t0 = time.time()
    decoded_bytes = b""

    while sent < max_send:
        pkt = enc.next_packet(); sent += 1
        truth_payload[pkt_key(pkt)] = pkt.payload

        if rng.random() < loss:
            drop_loss += 1
            continue

        tx_bytes = pack_packet_to_bytes(pkt)
        rx_bytes = bits_to_bytes(flip_bits(bytes_to_bits(tx_bytes), ber, rng))

        try:
            rpkt = unpack_packet_from_bytes(rx_bytes)
        except Exception:
            drop_hdr += 1
            continue

        if not allow_payload_errors:
            if truth_payload.get(pkt_key(rpkt), b"") != rpkt.payload:
                drop_payload += 1
                continue

        recvd_valid += 1
        dec.add_packet(rpkt)
        if dec.is_decoded():
            decoded_bytes = dec.reconstruct()
            ok = (decoded_bytes == message)
            dt = time.time() - t0
            s_hat = recvd_valid / float(sent)
            e_tx_hat = recvd_valid / max(s_hat, 1e-9)
            return {
                "ok": ok, "decoded": decoded_bytes,
                "k": k, "block_size": block_size, "msg_len": len(message),
                "loss": loss, "ber": ber, "mode": mode, "overhead": overhead if mode=="fixed" else None,
                "sent": sent, "received_valid": recvd_valid,
                "rx_over_k": recvd_valid/float(k), "tx_over_k": sent/float(k),
                "drop_loss": drop_loss, "drop_hdr": drop_hdr, "drop_payload": drop_payload,
                "s_hat": s_hat, "E_tx_hat": e_tx_hat, "time_sec": dt
            }

        if mode == "fixed" and sent >= max_send:
            break

    dt = time.time() - t0
    s_hat = (recvd_valid / float(sent)) if sent>0 else 0.0
    e_tx_hat = (recvd_valid / s_hat) if s_hat>0 else float('inf')
    return {
        "ok": False, "decoded": b"",
        "k": k, "block_size": block_size, "msg_len": len(message),
        "loss": loss, "ber": ber, "mode": mode, "overhead": overhead if mode=="fixed" else None,
        "sent": sent, "received_valid": recvd_valid,
        "rx_over_k": recvd_valid/float(k) if k else 0.0, "tx_over_k": sent/float(k) if k else 0.0,
        "drop_loss": drop_loss, "drop_hdr": drop_hdr, "drop_payload": drop_payload,
        "s_hat": s_hat, "E_tx_hat": e_tx_hat, "time_sec": dt
    }

def pretty(s: dict) -> str:
    badge = "[OK]" if s["ok"] else "[FAIL]"
    lines = [
        f"{badge} mode={s['mode']}  loss={s['loss']:.2f}  ber={s['ber']:.4f}  k={s['k']}  block_size={s['block_size']}  msg_len={s['msg_len']}"
    ]
    if s["mode"]=="fixed": lines.append(f"     overhead={s['overhead']:.2f}")
    lines += [
        f"     sent={s['sent']}  received(valid)={s['received_valid']}  rx/k={s['rx_over_k']:.3f}  tx/k={s['tx_over_k']:.3f}  time={s['time_sec']:.3f}s",
        f"     drops: loss={s['drop_loss']}  hdr_fail={s['drop_hdr']}  payload_fail={s['drop_payload']}",
        f"     observed success rate ŝ = {s['s_hat']:.3f}  ⇒ Ê[tx | R] ≈ {s['E_tx_hat']:.2f}",
    ]
    if s["ok"]:
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
    ap.add_argument("--message", help=f"消息文件路径（默认：{DEF_MSG_PATH}）")
    ap.add_argument("--block-size", type=int, default=0, help="LT 源块大小；0=自动选择")
    ap.add_argument("--k-min", type=int, default=8, help="自动选择目标 k 的下界")
    ap.add_argument("--k-max", type=int, default=64, help="自动选择目标 k 的上界")
    ap.add_argument("--allowed-sizes", default="8,16,32,64,128,256,512,1024", help="允许的 block_size 列表（字节），逗号分隔")
    ap.add_argument("--base-seed", type=int, default=1234)
    ap.add_argument("--loss", type=float, default=0.3, help="丢包率")
    ap.add_argument("--ber", type=float, default=0.0, help="位翻转概率（模拟隐写误码）")
    ap.add_argument("--mode", choices=["until-success","fixed"], default="until-success")
    ap.add_argument("--overhead", type=float, default=0.2)
    ap.add_argument("--max-send-factor", type=float, default=10.0)
    ap.add_argument("--rng-seed", type=int, default=20240829)
    ap.add_argument("--allow-payload-errors", action="store_true", help="允许错误 payload 进入 LT 解码（默认丢弃）")
    args = ap.parse_args()

    # 读取消息（默认 data/samples/msg.txt）
    msg_path = args.message or DEF_MSG_PATH
    if not os.path.isfile(msg_path):
        raise FileNotFoundError(f"找不到消息文件：{msg_path}")
    with open(msg_path, "rb") as f: msg = f.read()

    # 自动/手动 block_size
    if args.block_size and args.block_size > 0:
        bs = int(args.block_size)
        k = max(1, (len(msg) + bs - 1) // bs)
        print(f"[INFO] manual block_size={bs} -> k={k} (msg_len={len(msg)})")
    else:
        allowed = tuple(int(x) for x in args.allowed_sizes.split(",") if x.strip())
        bs, k = choose_block_size_auto(len(msg), args.k_min, args.k_max, allowed)
        print(f"[AUTO] chose block_size={bs} (allowed={allowed}) for msg_len={len(msg)} -> k={k} (target {args.k_min}..{args.k_max})")

    stats = run_once(
        message=msg, block_size=bs, base_seed=args.base_seed,
        loss=args.loss, ber=args.ber, mode=args.mode, overhead=args.overhead,
        max_send_factor=args.max_send_factor, rng_seed=args.rng_seed,
        allow_payload_errors=args.allow_payload_errors
    )
    print(pretty(stats))

if __name__ == "__main__":
    main()
