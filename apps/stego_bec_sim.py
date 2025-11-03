# apps/stego_bec_sim.py
# -*- coding: utf-8 -*-
"""
最小喷泉码×图像嵌入×丢包(BEC)模拟器（无鲁棒性，仅LSB，PNG）
- 支持配置文件：--config config/stego.json（common + stego_bec 段），命令行覆盖配置
- 每个喷泉包 -> 一张PNG：在Y通道的LSB顺序写入一个小容器(PKT0)
- 可用 --loss 随机丢弃，也可用 --keep 指定保留区间（如 20-80,100-120）
- --gen-only 只生成帧不解码
"""
from __future__ import annotations
import os, sys, argparse, struct, zlib, time, random, json, hashlib
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image

# ------- 兼容导入 LT 实现 -------
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'fountain'), HERE]
try:
    from fountain.lt_min import LTEncoder, LTDecoder  # type: ignore
except Exception:
    from lt_min import LTEncoder, LTDecoder  # type: ignore

MAGIC = b'PKT0'  # 简单包容器标识
HDR_FMT = "<4sBBIQI"  # MAGIC, typ, rsv, sys_idx(u32), seed(u64), payload_len(u32)
HDR_SIZE = struct.calcsize(HDR_FMT)  # 22字节

@dataclass
class SimplePacket:
    payload: bytes
    sys_idx: Optional[int] = None
    seed: Optional[int] = None

# ------------------ 工具：配置 / 区间解析 ------------------
def load_config(cfg_path: str, section: str) -> dict:
    """从 JSON 加载配置：合并 common + 指定 section；不存在返回空"""
    if not cfg_path or not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = {}
    cfg.update(data.get("common", {}))
    cfg.update(data.get(section, {}))
    return cfg

def apply_cfg(args: argparse.Namespace, cfg: dict, keys: list[str]):
    """把 cfg 里的键填到 args（仅填充仍为 None 的项）；命令行优先"""
    for k in keys:
        if getattr(args, k, None) is None and k in cfg:
            setattr(args, k, cfg[k])

def parse_keep_spec(spec: str, total: int) -> List[int]:
    """解析 '20-80,100-120,5,9' → 有序唯一索引列表（0..total-1）"""
    keep = set()
    spec = (spec or "").strip().replace(" ", "")
    if not spec:
        return []
    for part in spec.split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a) if a != "" else 0
            b = int(b) if b != "" else (total - 1)
            a = max(0, min(total - 1, a))
            b = max(0, min(total - 1, b))
            if a > b:
                a, b = b, a
            keep.update(range(a, b + 1))
        else:
            i = int(part)
            if 0 <= i < total:
                keep.add(i)
    return sorted(keep)

# ------------------ 工具：容器编解码 ------------------
def serialize_pkt(pkt) -> bytes:
    """LT 包对象 -> 容器字节串：MAGIC|typ|sys_idx|seed|len|payload|CRC32"""
    sys_idx = getattr(pkt, "sys_idx", None)
    seed = getattr(pkt, "seed", None)
    typ = 0 if sys_idx is not None else 1
    sys_idx_u32 = int(sys_idx) if sys_idx is not None else 0xFFFFFFFF
    seed_u64 = int(seed) if seed is not None else 0
    payload = pkt.payload
    hdr = struct.pack(HDR_FMT, MAGIC, typ, 0, sys_idx_u32, seed_u64, len(payload))
    body = hdr + payload
    crc = struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
    return body + crc

def parse_pkt(buf: bytes) -> SimplePacket:
    """容器字节串 -> SimplePacket；CRC 不符抛错"""
    if len(buf) < HDR_SIZE + 4:
        raise ValueError("buffer too short")
    magic, typ, _rsv, sys_idx_u32, seed_u64, paylen = struct.unpack(HDR_FMT, buf[:HDR_SIZE])
    if magic != MAGIC:
        raise ValueError("bad magic")
    total = HDR_SIZE + paylen
    if len(buf) < total + 4:
        raise ValueError("buffer truncated")
    body = buf[:total]
    crc_ok = (zlib.crc32(body) & 0xFFFFFFFF) == struct.unpack("<I", buf[total:total+4])[0]
    if not crc_ok:
        raise ValueError("CRC mismatch")
    payload = buf[HDR_SIZE:total]
    if typ == 0:
        return SimplePacket(payload=payload, sys_idx=int(sys_idx_u32), seed=None)
    else:
        return SimplePacket(payload=payload, sys_idx=None, seed=int(seed_u64))

# ------------------ 工具：Y通道 LSB 嵌入/提取 ------------------
def img_to_y_array(img: Image.Image) -> np.ndarray:
    if img.mode != "YCbCr":
        img = img.convert("YCbCr")
    Y, Cb, Cr = img.split()
    return np.array(Y, dtype=np.uint8)

def y_array_to_img(y: np.ndarray, tmpl: Image.Image) -> Image.Image:
    if tmpl.mode != "YCbCr":
        tmpl = tmpl.convert("YCbCr")
    Y0, Cb, Cr = tmpl.split()
    Y1 = Image.fromarray(y.astype(np.uint8), mode="L")
    return Image.merge("YCbCr", (Y1, Cb, Cr)).convert("RGB")  # 存PNG用RGB

def embed_bytes_into_image(cover: Image.Image, data: bytes) -> Image.Image:
    y = img_to_y_array(cover)
    H, W = y.shape
    cap_bits = H * W
    need_bits = len(data) * 8
    if need_bits > cap_bits:
        raise ValueError(f"容量不足：需要 {need_bits} bit，但只有 {cap_bits} bit（{H}x{W}）")
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    y_flat = y.reshape(-1).copy()
    y_flat[:need_bits] = (y_flat[:need_bits] & 0xFE) | bits
    y2 = y_flat.reshape(H, W)
    return y_array_to_img(y2, cover)

def extract_bytes_from_image(stego: Image.Image) -> bytes:
    y = img_to_y_array(stego)
    y_flat = y.reshape(-1)
    need_bits_hdr = HDR_SIZE * 8
    if need_bits_hdr > y_flat.size:
        raise ValueError("图像太小，连头都放不下")
    bits_hdr = (y_flat[:need_bits_hdr] & 1).astype(np.uint8)
    hdr_bytes = np.packbits(bits_hdr).tobytes()
    try:
        magic, typ, _rsv, sys_idx_u32, seed_u64, paylen = struct.unpack(HDR_FMT, hdr_bytes)
    except struct.error:
        raise ValueError("读头失败（结构不符）")
    if magic != MAGIC:
        raise ValueError("MAGIC 不匹配")
    total_len = HDR_SIZE + paylen + 4
    need_bits_total = total_len * 8
    if need_bits_total > y_flat.size:
        raise ValueError("图像容量不够，容器被截断")
    bits_all = (y_flat[:need_bits_total] & 1).astype(np.uint8)
    all_bytes = np.packbits(bits_all).tobytes()
    return all_bytes

# ------------------ 其它工具 ------------------
def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def ensure_dir_of(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ------------------ 主流程 ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "stego.json"), help="配置文件路径（JSON）")
    # 留空让配置来填，命令行覆盖
    ap.add_argument("--message", default=None, help="要传送的文件路径（任意类型）")
    ap.add_argument("--cover", default=None, help="封面图路径（PNG/JPG 等；会被重复使用）")
    ap.add_argument("--n", type=int, default=None, help="生成包/帧的数量")
    ap.add_argument("--loss", type=float, default=None, help="随机丢帧概率（0~1）")
    ap.add_argument("--keep", default=None, help="保留帧索引，如 20-80,100-120；留空则按 --loss")
    ap.add_argument("--block-size", type=int, default=None, help="LT 源块大小")
    ap.add_argument("--base-seed", type=int, default=None, help="LT/随机丢弃的基础种子")
    ap.addArgument = ap.add_argument  # alias
    ap.add_argument("--frames-dir", default=None, help="输出图像目录")
    ap.add_argument("--out", default=None, help="恢复成功后的落盘路径")
    ap.add_argument("--gen-only", dest="gen_only", action="store_true")
    ap.add_argument("--no-gen-only", dest="gen_only", action="store_false")
    ap.set_defaults(gen_only=None)
    ap.add_argument("--preview", dest="preview", action="store_true")
    ap.add_argument("--no-preview", dest="preview", action="store_false")
    ap.set_defaults(preview=None)
    args = ap.parse_args()

    # 加载配置
    cfg = load_config(args.config, "stego_bec")
    apply_cfg(args, cfg, ["message","cover","n","loss","keep","block_size","base_seed",
                          "frames_dir","out","gen_only","preview"])

    # 兜底默认
    if args.block_size is None: args.block_size = 512
    if args.base_seed  is None: args.base_seed  = 1234
    if args.n          is None: args.n          = 120
    if args.loss       is None: args.loss       = 0.3
    if args.keep       is None: args.keep       = ""
    if args.frames_dir is None: args.frames_dir = "runs/frames_bec"
    if args.out        is None: args.out        = "runs/recovered.bin"
    if args.gen_only   is None: args.gen_only   = False
    if args.preview    is None: args.preview    = False

    # 读消息 + 封面
    if not args.message or not os.path.isfile(args.message):
        raise FileNotFoundError(f"找不到消息文件：{args.message}")
    with open(args.message, "rb") as f:
        msg = f.read()
    if len(msg) == 0:
        raise ValueError("消息为空")
    cover = Image.open(args.cover).convert("RGB") if args.cover else None
    if cover is None:
        raise FileNotFoundError(f"找不到封面图：{args.cover}")

    # LT 编解码器
    enc = LTEncoder(msg, block_size=args.block_size, base_seed=args.base_seed, systematic=True)
    dec = LTDecoder()

    # 生成帧
    os.makedirs(args.frames_dir, exist_ok=True)
    frames: List[str] = []
    t0 = time.time()
    for i in range(args.n):
        pkt = enc.next_packet()
        blob = serialize_pkt(pkt)
        stego = embed_bytes_into_image(cover, blob)
        path = os.path.join(args.frames_dir, f"frame_{i:04d}.png")
        stego.save(path, format="PNG", optimize=False)
        frames.append(path)
    gen_elapsed = time.time() - t0

    print(f"[GEN] frames={len(frames)}  dir={os.path.abspath(args.frames_dir)}  time={gen_elapsed:.3f}s")

    if args.gen_only:
        return  # 只生成，不解码

    # 选择保留帧：优先 --keep，其次按 --loss 随机
    if args.keep.strip():
        keep_idx = parse_keep_spec(args.keep, len(frames))
        kept = [frames[i] for i in keep_idx]
        drops = len(frames) - len(kept)
        print(f"[KEEP] 指定保留 {len(kept)} 帧；丢弃 {drops}")
    else:
        rng = random.Random(args.base_seed ^ 0xDEADBEEF)
        kept = [p for p in frames if rng.random() >= args.loss]
        drops = len(frames) - len(kept)
        print(f"[BEC ] 随机丢弃 {drops}/{len(frames)}（loss={args.loss:.3f}）")

    # 译码
    rx = 0
    for p in kept:
        try:
            all_bytes = extract_bytes_from_image(Image.open(p))
            sp = parse_pkt(all_bytes)
            dec.add_packet(sp)
            rx += 1
            if dec.is_decoded():
                break
        except Exception:
            pass  # 读取失败等同擦除

    # 结果
    k = max(1, (len(msg) + args.block_size - 1)//args.block_size)
    print("\n========== RESULT ==========")
    print(f"k={k}  block_size={args.block_size}  msg_len={len(msg)}")
    print(f"frames(total)={len(frames)}  kept={len(kept)}  drops={drops}")
    print(f"received(valid)={rx}  rx/k≈{rx/float(k):.3f}")

    if dec.is_decoded():
        rec = dec.reconstruct()
        ensure_dir_of(args.out)
        with open(args.out, "wb") as f:
            f.write(rec)
        ok = (rec == msg)
        print(f"[OK] Recovered. Bytes match: {ok}")
        print(f"src_sha256={sha256_hex(msg)}")
        print(f"rec_sha256={sha256_hex(rec)}")
        print(f"→ 写出: {args.out}")

        if args.preview:
            base = os.path.splitext(args.out)[0]
            try:
                ab = os.path.abspath
                if rec.startswith(b"\x89PNG\r\n\x1a\n"):
                    p = ab(base + ".png"); open(p,"wb").write(rec); print(f"[preview] PNG -> {p}")
                    if os.name == "nt": os.startfile(p)
                elif rec.startswith(b"\xff\xd8"):
                    p = ab(base + ".jpg"); open(p,"wb").write(rec); print(f"[preview] JPEG -> {p}")
                    if os.name == "nt": os.startfile(p)
                else:
                    head = rec[:800]
                    try:
                        s = head.decode("utf-8", errors="replace")
                        print("[preview] 文本前800字：\n"+s)
                    except Exception:
                        print("[preview] 非文本，前64字节：\n"+head.hex(" "))
            except Exception as e:
                print(f"[preview] 打开预览失败：{e}")
    else:
        print("[FAIL] 未能在保留的帧内解码成功；可调大 n、降低 loss 或用 --keep 指定更多帧。")

if __name__ == "__main__":
    main()
