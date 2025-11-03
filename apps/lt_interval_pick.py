# -*- coding: utf-8 -*-
"""
lt_interval_pick.py  (config 版，含绝对路径预览)
纯 LT：生成 N 个包，仅用【保留区间】的包进行译码；可导出 payload；成功后写盘/预览。
支持配置文件：--config config/lt.json
"""
from __future__ import annotations
import os, sys, argparse, hashlib, json
from typing import List, Tuple

# ---------- 兼容导入 LT 实现 ----------
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'fountain'), HERE]
try:
    from fountain.lt_min import LTEncoder, LTDecoder, LTPacket
except Exception:
    from lt_min import LTEncoder, LTDecoder, LTPacket  # type: ignore

# ---------- 工具 ----------
def choose_block_size_auto(msg_len: int,
                           k_min: int = 8, k_max: int = 64,
                           allowed_sizes = (8,16,32,64,128,256,512,1024)) -> Tuple[int,int]:
    assert msg_len > 0
    cand = []
    for bs in allowed_sizes:
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

def parse_keep_spec(spec: str, total: int) -> List[int]:
    keep = set()
    spec = (spec or "").strip().replace(' ', '')
    if not spec:
        return []
    for part in spec.split(','):
        if not part: continue
        if '-' in part:
            a, b = part.split('-', 1)
            a = int(a) if a != '' else 0
            b = int(b) if b != '' else (total - 1)
            a = max(0, min(total - 1, a))
            b = max(0, min(total - 1, b))
            if a > b: a, b = b, a
            keep.update(range(a, b + 1))
        else:
            i = int(part)
            if 0 <= i < total: keep.add(i)
    return sorted(keep)

def hex_preview(b: bytes, n: int = 32) -> str:
    return b[:n].hex(' ')

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def dump_packet_payloads(outdir: str, keep_idx: List[int], pkts: List['LTPacket']) -> None:
    os.makedirs(outdir, exist_ok=True)
    manifest_lines = ["idx,type,sys_idx,seed,bytes,file"]
    for i in keep_idx:
        p = pkts[i]
        kind = "SYS" if getattr(p, "sys_idx", None) is not None else "CODE"
        fname = f"pkt_{i:04d}_{kind.lower()}.bin"
        path = os.path.join(outdir, fname)
        with open(path, 'wb') as f:
            f.write(p.payload)
        manifest_lines.append(f"{i},{kind},{p.sys_idx if getattr(p,'sys_idx',None) is not None else ''},"
                              f"{p.seed if getattr(p,'seed',None) is not None else ''},{len(p.payload)},{fname}")
    with open(os.path.join(outdir, "manifest.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines))

def detect_and_preview(rec: bytes, out_path_base: str) -> None:
    """绝对路径预览；将图片文件另存为正确后缀并尝试打开。"""
    def _is_png(b):  return b.startswith(b"\x89PNG\r\n\x1a\n")
    def _is_jpg(b):  return b.startswith(b"\xff\xd8")
    def _is_gif(b):  return b.startswith(b"GIF87a") or b.startswith(b"GIF89a")
    def _is_bmp(b):  return b.startswith(b"BM")
    def _is_webp(b): return b.startswith(b"RIFF") and (len(b) >= 12 and b[8:12] == b"WEBP")
    def _is_text(b):
        import string
        PRINT = set(bytes(string.printable, "ascii"))
        if not b: return False
        okc = sum(1 for x in b if x in PRINT)
        return okc / len(b) > 0.9
    try:
        base_abs = os.path.abspath(out_path_base)
        def _save_and_open(ext: str):
            path = base_abs + ext
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fw:
                fw.write(rec)
            print(f"[preview] 识别为 {ext.upper().lstrip('.')} -> {path}")
            try:
                if os.name == "nt":
                    os.startfile(path)
            except Exception as ee:
                print(f"[preview] 打开失败：{ee}\n你也可以手动运行：start \"\" \"{path}\"")

        if   _is_png(rec):  _save_and_open(".png")
        elif _is_jpg(rec):  _save_and_open(".jpg")
        elif _is_gif(rec):  _save_and_open(".gif")
        elif _is_bmp(rec):  _save_and_open(".bmp")
        elif _is_webp(rec): _save_and_open(".webp")
        elif _is_text(rec):
            head = rec[:800].decode("utf-8", errors="replace")
            print("[preview] 识别为文本，前 800 字符：\n" + head)
        else:
            print("[preview] 未识别的二进制类型，前 64 字节十六进制：")
            print(rec[:64].hex(" "))
    except Exception as e:
        print(f"[preview] 打开预览失败：{e}")

def load_config(cfg_path: str, section: str) -> dict:
    if not cfg_path or not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = {}
    cfg.update(data.get("common", {}))
    cfg.update(data.get(section, {}))
    return cfg

def apply_cfg(args: argparse.Namespace, cfg: dict, keys: list[str]):
    for k in keys:
        if getattr(args, k, None) is None and k in cfg:
            setattr(args, k, cfg[k])

# ---------- 主流程 ----------
def main():
    DEF_MSG = os.path.normpath(os.path.join(ROOT, 'data', 'samples', 'msg.txt'))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "lt.json"), help="配置文件路径（JSON）")
    ap.add_argument("--message", default=None, help="消息文件（任意类型）")
    ap.add_argument("--block-size", type=int, default=None, help="源块大小；None 或 0=自动选择")
    ap.add_argument("--k-min", type=int, default=None)
    ap.add_argument("--k-max", type=int, default=None)
    ap.add_argument("--allowed-sizes", default=None, help="逗号分隔或 JSON 列表")
    ap.add_argument("--base-seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=None, help="总生成包数")
    ap.add_argument("--keep", default=None, help="保留区间，如 '20-80' 或 '0-15,50-99'")
    ap.add_argument("--dump-dir", default=None, help="若指定，把保留区间内每个包的 payload 落盘")
    ap.add_argument("--show-full", dest="show_full", action="store_true")
    ap.add_argument("--no-show-full", dest="show_full", action="store_false")
    ap.set_defaults(show_full=None)
    ap.add_argument("--out", default=None, help="成功译码后落盘路径")
    ap.add_argument("--preview", dest="preview", action="store_true")
    ap.add_argument("--no-preview", dest="preview", action="store_false")
    ap.set_defaults(preview=None)
    args = ap.parse_args()

    cfg = load_config(args.config, "interval")
    apply_cfg(args, cfg, ["message","block_size","k_min","k_max","allowed_sizes",
                          "base_seed","n","keep","dump_dir","show_full","out","preview"])

    # 兜底默认
    if args.message is None: args.message = DEF_MSG
    if args.block_size in (None,): args.block_size = 0
    if args.k_min is None: args.k_min = 8
    if args.k_max is None: args.k_max = 64
    if args.allowed_sizes is None:
        allowed = (8,16,32,64,128,256,512,1024)
    else:
        if isinstance(args.allowed_sizes, str):
            if args.allowed_sizes.strip().startswith("["):
                allowed = tuple(json.loads(args.allowed_sizes))
            else:
                allowed = tuple(int(x) for x in args.allowed_sizes.split(",") if x.strip())
        else:
            allowed = tuple(args.allowed_sizes)
    if args.base_seed is None: args.base_seed = 1234
    if args.n is None: args.n = 100
    if args.keep is None: args.keep = ""
    if args.dump_dir is None: args.dump_dir = ""
    if args.show_full is None: args.show_full = False
    if args.out is None: args.out = "runs/interval_decoded.bin"
    if args.preview is None: args.preview = False

    # 读消息
    if not os.path.isfile(args.message):
        raise FileNotFoundError(f"找不到消息文件：{args.message}")
    with open(args.message, 'rb') as f:
        msg = f.read()
    if len(msg) == 0:
        raise ValueError("消息为空")

    # 选 block_size
    if args.block_size and args.block_size > 0:
        bs = int(args.block_size)
        k = max(1, (len(msg) + bs - 1) // bs)
        print(f"[INFO] manual block_size={bs} -> k={k} (msg_len={len(msg)})")
    else:
        bs, k = choose_block_size_auto(len(msg), args.k_min, args.k_max, allowed)
        print(f"[AUTO] chose block_size={bs} (allowed={allowed}) for msg_len={len(msg)} -> k={k} (target {args.k_min}..{args.k_max})")

    # 生成 N 个包
    N = int(args.n)
    enc = LTEncoder(msg, block_size=bs, base_seed=args.base_seed, systematic=True)
    pkts: List['LTPacket'] = [enc.next_packet() for _ in range(N)]
    print(f"\n[GEN] 总生成包数 N={N}；系统包索引范围：0..{k-1}（若 N>k，则 k..{N-1} 为编码包）")

    # 选择保留区间
    keep_spec = args.keep.strip() if isinstance(args.keep, str) else ""
    if not keep_spec:
        keep_spec = input("请输入【保留】的区间（如 20-80 或 0-15,50-99；支持 -b / a-）： ").strip()
    keep_idx = parse_keep_spec(keep_spec, N)
    if not keep_idx:
        print("[WARN] 你没有选择任何包，无法译码。")
        return

    # 展示被保留包的payload信息
    print("\n========== SELECTED PACKETS (RAW PAYLOAD) ==========")
    print(" idx | type | sys_idx | seed       | bytes | payload hex preview")
    print("-----+------+---------+------------+-------+------------------------------")
    for i in keep_idx:
        p = pkts[i]
        kind = "SYS" if getattr(p,"sys_idx",None) is not None else "CODE"
        sid  = f"{p.sys_idx}" if getattr(p,"sys_idx",None) is not None else "-"
        sseed= f"{p.seed}" if getattr(p,"seed",None) is not None else "-"
        hexv = p.payload.hex(' ') if args.show_full else hex_preview(p.payload, 32)
        print(f"{i:4d} | {kind:4s} | {sid:7s} | {sseed:10s} | {len(p.payload):5d} | {hexv}")

    if args.dump_dir:
        dump_packet_payloads(args.dump_dir, keep_idx, pkts)
        print(f"\n[WRITE] 已写入 {len(keep_idx)} 个 payload 到目录：{args.dump_dir}（含 manifest.csv）")

    # 仅用保留区间包进行译码
    dec = LTDecoder()
    kept_sys = kept_coded = 0
    for i in keep_idx:
        pkt = pkts[i]
        if getattr(pkt, "sys_idx", None) is not None: kept_sys += 1
        else: kept_coded += 1
        dec.add_packet(pkt)

    ok = dec.is_decoded()
    rx = len(keep_idx)
    print("\n========== RESULT ==========")
    print(f"k={k}  block_size={bs}  msg_len={len(msg)}")
    print(f"选择区间：{keep_spec}")
    print(f"保留数量={rx}  (sys={kept_sys}, coded={kept_coded})  约等于 rx/k={rx/float(k):.3f}")
    print(f"是否成功恢复：{'YES' if ok else 'NO'}")

    if ok:
        rec = dec.reconstruct()
        # 自动创建输出目录
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "wb") as f:
            f.write(rec)

        same = (rec == msg)
        print(f"[OK] 写出：{args.out}  Bytes match: {same}")
        print(f"src_sha256={sha256_hex(msg)}")
        print(f"rec_sha256={sha256_hex(rec)}")

        if args.preview:
            base = os.path.splitext(args.out)[0]
            detect_and_preview(rec, base)
    else:
        print("建议：扩大保留区间（尤其包含部分系统包），或增加总生成包数 N。")

if __name__ == "__main__":
    main()
