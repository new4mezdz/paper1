# -*- coding: utf-8 -*-
"""
stego_interval_pick.py
同一张封面反复写入喷泉包 -> 生成 N 帧 -> 交互/参数选择“保留区间” -> 仅用这些帧尝试恢复
嵌入方式：RGB 的 R 通道 LSB（PNG 无损）
"""
from __future__ import annotations
import os, sys, argparse, json, struct, zlib, hashlib
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from PIL import Image

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'fountain'), HERE]
try:
    from fountain.lt_min import LTEncoder, LTDecoder  # type: ignore
except Exception:
    from lt_min import LTEncoder, LTDecoder  # type: ignore

from fountain.auto_blocksize import resolve_block_size, AutoBSResult

MAGIC = b'PKT0'
HDR_FMT = "<4sBBIQI"
HDR_SIZE = struct.calcsize(HDR_FMT)

@dataclass
class SimplePacket:
    payload: bytes
    sys_idx: Optional[int] = None
    seed: Optional[int] = None
    k: Optional[int] = None
    block_size: Optional[int] = None
    base_seed: Optional[int] = None
    msg_len: Optional[int] = None
    @property
    def data(self): return self.payload
    @property
    def buf(self): return self.payload
    @property
    def index(self): return self.sys_idx
    @property
    def sys_index(self): return self.sys_idx
    @property
    def is_systematic(self): return self.sys_idx is not None
    @property
    def message_length(self): return self.msg_len
    @property
    def total_length(self): return self.msg_len

# ---------- 配置/工具 ----------
def load_cfg(path: str, section: str) -> dict:
    if not os.path.isfile(path): return {}
    with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    cfg = {}; cfg.update(data.get("common", {})); cfg.update(data.get(section, {})); return cfg

def apply_cfg(args, cfg, keys):
    for k in keys:
        if getattr(args, k, None) is None and k in cfg: setattr(args, k, cfg[k])

def parse_keep(spec: str, total: int) -> List[int]:
    keep=set(); s=(spec or "").strip().replace(" ","")
    if not s: return []
    for part in s.split(","):
        if not part: continue
        if "-" in part:
            a,b=part.split("-",1)
            a=int(a) if a!="" else 0
            b=int(b) if b!="" else (total-1)
            a=max(0,min(total-1,a)); b=max(0,min(total-1,b))
            if a>b: a,b=b,a
            keep.update(range(a,b+1))
        else:
            i=int(part)
            if 0<=i<total: keep.add(i)
    return sorted(keep)

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def serialize_pkt(pkt) -> bytes:
    sys_idx = getattr(pkt, "sys_idx", None)
    seed = getattr(pkt, "seed", None)
    typ = 0 if sys_idx is not None else 1
    sys_idx_u32 = int(sys_idx) if sys_idx is not None else 0xFFFFFFFF
    seed_u64 = int(seed) if seed is not None else 0
    payload = getattr(pkt, "payload", None) or getattr(pkt, "data", None) or getattr(pkt, "buf", None)
    if payload is None: raise ValueError("pkt 缺少 payload/data/buf 字段")
    hdr = struct.pack(HDR_FMT, MAGIC, typ, 0, sys_idx_u32, seed_u64, len(payload))
    body = hdr + payload
    crc = struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
    return body + crc

def parse_pkt(buf: bytes) -> SimplePacket:
    if len(buf) < HDR_SIZE+4: raise ValueError("short")
    magic, typ, _rsv, sys_idx_u32, seed_u64, paylen = struct.unpack(HDR_FMT, buf[:HDR_SIZE])
    if magic != MAGIC: raise ValueError("bad magic")
    total = HDR_SIZE + paylen
    if len(buf) < total+4: raise ValueError("truncated")
    body = buf[:total]
    if (zlib.crc32(body)&0xFFFFFFFF) != struct.unpack("<I", buf[total:total+4])[0]:
        raise ValueError("CRC mismatch")
    payload = buf[HDR_SIZE:total]
    if typ == 0: return SimplePacket(payload=payload, sys_idx=int(sys_idx_u32))
    else:        return SimplePacket(payload=payload, seed=int(seed_u64))

# ---------- R 通道 LSB ----------
def img_to_r(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return arr[..., 0]

def r_to_img(r: np.ndarray, tmpl: Image.Image) -> Image.Image:
    arr = np.array(tmpl.convert("RGB"), dtype=np.uint8)
    arr[..., 0] = r.astype(np.uint8)
    return Image.fromarray(arr)

def embed_bytes(cover: Image.Image, data: bytes) -> Image.Image:
    r = img_to_r(cover); H, W = r.shape
    need = len(data)*8
    cap = H*W
    if need > cap: raise ValueError(f"容量不足：需要 {need} bit，但只有 {cap} bit（{H}x{W}）")
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    flat = r.reshape(-1).copy()
    flat[:need] = (flat[:need] & 0xFE) | bits
    return r_to_img(flat.reshape(H,W), cover)

def extract_bytes(stego: Image.Image) -> bytes:
    r = img_to_r(stego); flat = r.reshape(-1)
    need_hdr = HDR_SIZE*8
    if need_hdr > flat.size: raise ValueError("图像太小，连头都放不下")
    bits_hdr = (flat[:need_hdr] & 1).astype(np.uint8)
    hdr_bytes = np.packbits(bits_hdr).tobytes()
    try:
        magic, typ, _rsv, sys_idx_u32, seed_u64, paylen = struct.unpack(HDR_FMT, hdr_bytes)
    except struct.error:
        raise ValueError("读头失败（结构不符）")
    if magic != MAGIC: raise ValueError("MAGIC 不匹配")
    total_len = HDR_SIZE + paylen + 4
    need_total = total_len*8
    if need_total > flat.size: raise ValueError("图像容量不够，容器被截断")
    bits_all = (flat[:need_total] & 1).astype(np.uint8)
    return np.packbits(bits_all).tobytes()

# ---------- 主 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "stego.json"))
    ap.add_argument("--message", default=None)
    ap.add_argument("--cover", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--keep", default=None, help="如 20-120,0-15,50-99；支持 -b / a-")
    ap.add_argument("--show-full", dest="show_full", action="store_true")
    ap.add_argument("--no-show-full", dest="show_full", action="store_false")
    ap.set_defaults(show_full=None)
    ap.add_argument("--block-size", default=None)
    ap.add_argument("--force-k", type=int, default=None)  # ← 新增
    ap.add_argument("--target-k", type=int, default=None)
    ap.add_argument("--headroom", type=float, default=None)
    ap.add_argument("--base-seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--systematic", dest="systematic", action="store_true")
    ap.add_argument("--no-systematic", dest="systematic", action="store_false")
    ap.set_defaults(systematic=None)
    ap.add_argument("--preview", dest="preview", action="store_true")
    ap.add_argument("--no-preview", dest="preview", action="store_false")
    ap.set_defaults(preview=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config, "stego_interval")
    apply_cfg(args, cfg, ["message","cover","n","frames_dir","keep","show_full",
                          "block_size","base_seed","out","preview","systematic",
                          "target_k","headroom","force_k"])

    if args.base_seed  is None: args.base_seed  = 1234
    if args.n          is None: args.n          = 500
    if args.frames_dir is None: args.frames_dir = "runs/frames_interval"
    if args.keep       is None: args.keep       = ""
    if args.show_full  is None: args.show_full  = False
    if args.out        is None: args.out        = "runs/interval_recovered.bin"
    if args.systematic is None: args.systematic = True
    if args.preview    is None: args.preview    = True
    if args.target_k   is None: args.target_k   = 200
    if args.headroom   is None: args.headroom   = 0.98

    if not args.message or not os.path.isfile(args.message): raise FileNotFoundError(args.message)
    with open(args.message, "rb") as f: msg = f.read()
    cover = Image.open(args.cover).convert("RGB") if args.cover else None
    if cover is None: raise FileNotFoundError(str(args.cover))

    overhead = HDR_SIZE + 4
    res: AutoBSResult = resolve_block_size(
        arg_block_size=args.block_size,
        msg_len=len(msg),
        cover_img=cover,
        target_k=args.target_k,
        headroom=args.headroom,
        overhead=overhead,
        bits_per_pixel=1,
        force_k=args.force_k,   # ← 关键
    )
    args.block_size = res.chosen_block_size
    k = max(1, (len(msg)+args.block_size-1)//args.block_size)

    if res.reason == "force_k":
        print(f"[FORCE] k={args.force_k} -> block_size={res.chosen_block_size} (k≈{res.k_estimate})")
    elif res.reason == "auto":
        print(f"[AUTO]  block_size={res.chosen_block_size}  "
              f"(target_k={args.target_k}, cap≈{res.frame_capacity_bytes}B, "
              f"max_bs≈{res.max_block_size}B, k≈{res.k_estimate})")
    else:
        print(f"[BS]    block_size={res.chosen_block_size}  "
              f"(cap≈{res.frame_capacity_bytes}B, max_bs≈{res.max_block_size}B, k≈{res.k_estimate})")

    # 生成 N 帧
    enc, dec = LTEncoder(msg, block_size=args.block_size, base_seed=args.base_seed, systematic=bool(args.systematic)), LTDecoder()
    os.makedirs(args.frames_dir, exist_ok=True)
    frames: List[str] = []
    for i in range(args.n):
        pkt = enc.next_packet()
        blob = serialize_pkt(pkt)
        stego = embed_bytes(cover, blob)
        path = os.path.join(args.frames_dir, f"frame_{i:04d}.png")
        stego.save(path, format="PNG", optimize=False)
        frames.append(path)

    print(f"[GEN] 生成 {len(frames)} 帧 -> {os.path.abspath(args.frames_dir)}（systematic={args.systematic}）")

    # 选择保留区间
    if not (args.keep or "").strip():
        args.keep = input("请输入【保留】的区间（如 20-120 或 0-15,50-99；支持 -b / a-）： ").strip()
    keep_idx = parse_keep(args.keep, len(frames))
    if not keep_idx:
        print("[WARN] 未选择任何帧，无法恢复。"); return

    # 展示摘要
    print("\n========== SELECTED FRAMES ==========")
    print(" idx | type | sys_idx | seed       | paylen | payload hex preview")
    print("-----+------+---------+------------+--------+------------------------------")
    kept_sys = kept_coded = 0
    for i in keep_idx:
        try:
            all_bytes = extract_bytes(Image.open(frames[i]))
            magic, typ, _rsv, sys_idx_u32, seed_u64, paylen = struct.unpack(HDR_FMT, all_bytes[:HDR_SIZE])
            kind = "SYS" if typ == 0 else "CODE"
            if kind == "SYS": kept_sys += 1
            else: kept_coded += 1
            head = all_bytes[HDR_SIZE:HDR_SIZE+min(24, paylen)].hex(' ') if not args.show_full else all_bytes[HDR_SIZE:HDR_SIZE+paylen].hex(' ')
            sid = f"{sys_idx_u32}" if typ==0 else "-"
            sseed = f"{seed_u64}" if typ!=0 else "-"
            print(f"{i:4d} | {kind:4s} | {sid:7s} | {sseed:10s} | {paylen:6d} | {head}")
        except Exception as e:
            print(f"{i:4d} | FAIL |   -     |    -       |    -   | {e}")

    # 只用保留帧尝试恢复
    for i in keep_idx:
        try:
            all_bytes = extract_bytes(Image.open(frames[i]))
            sp = parse_pkt(all_bytes)
            sp.k = k; sp.block_size = args.block_size; sp.base_seed = args.base_seed; sp.msg_len = len(msg)
            dec.add_packet(sp)
        except Exception:
            pass

    ok = dec.is_decoded()
    print("\n========== RESULT ==========")
    print(f"k={k}  block_size={args.block_size}  msg_len={len(msg)}")
    print(f"选择区间：{args.keep}")
    print(f"保留数量={len(keep_idx)}  (sys={kept_sys}, coded={kept_coded})  约等于 rx/k={len(keep_idx)/float(k):.3f}")
    print(f"是否成功恢复：{'YES' if ok else 'NO'}")

    if ok:
        rec = dec.reconstruct()
        out_abs = os.path.abspath(args.out); os.makedirs(os.path.dirname(out_abs), exist_ok=True)
        with open(out_abs, "wb") as f: f.write(rec)
        print(f"[OK] 写出：{out_abs}  Bytes match: {rec==msg}")
        print(f"src_sha256={sha256_hex(msg)}")
        print(f"rec_sha256={sha256_hex(rec)}")
        if args.preview:
            try:
                base = os.path.splitext(out_abs)[0]
                if rec.startswith(b"\x89PNG\r\n\x1a\n"):
                    p = base + ".png"; open(p,"wb").write(rec); print(f"[preview] PNG -> {p}")
                    if os.name=="nt": os.startfile(p)
                else:
                    head = rec[:800]
                    try:
                        s = head.decode("utf-8-sig", errors="replace")
                        print("[preview] 文本前800字：\n"+s)
                    except: print("[preview] 前64字节：\n"+head.hex(" "))
            except Exception as e:
                print(f"[preview] 失败：{e}")
    else:
        print("建议：扩大保留区间（尤其包含一部分 SYS 包），或增大 N。")

if __name__ == "__main__":
    main()
