# -*- coding: utf-8 -*-
"""
lt_until_success.py  (config 版，含绝对路径预览)
纯 LT：把任意文件当消息；独立丢包；解码成功即停（until-success）。
支持配置文件：--config config/lt.json
"""
from __future__ import annotations
import os, sys, time, argparse, hashlib, json
import numpy as np

# -------------------- 兼容导入 LT 实现 --------------------
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))     # 项目根
sys.path[:0] = [ROOT, os.path.join(ROOT, 'fountain'), HERE]
try:
    from fountain.lt_min import LTEncoder, LTDecoder
except Exception:
    from lt_min import LTEncoder, LTDecoder  # type: ignore

# -------------------- 工具函数 --------------------
def choose_block_size_auto(msg_len: int,
                           k_min: int = 8, k_max: int = 64,
                           allowed_sizes = (8,16,32,64,128,256,512,1024)) -> tuple[int,int]:
    assert msg_len > 0
    cand = []
    for bs in allowed_sizes:
        k = max(1, (msg_len + bs - 1) // bs)
        cand.append((bs, k))
    mid = 0.5 * (k_min + k_max)
    in_range = [(bs, k) for (bs, k) in cand if k_min <= k <= k_max]
    if in_range:
        in_range.sort(key=lambda x: (abs(x[1] - mid), -x[0]))
        return in_range[0]
    if all(k < k_min for _, k in cand):
        return min(cand, key=lambda x: x[0])
    return max(cand, key=lambda x: x[0])

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def load_config(cfg_path: str, section: str) -> dict:
    """从 JSON 加载：先合并 common，再合并指定 section。不存在时返回空。"""
    if not cfg_path or not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = {}
    cfg.update(data.get("common", {}))
    cfg.update(data.get(section, {}))
    return cfg

def apply_cfg(args: argparse.Namespace, cfg: dict, keys: list[str]):
    """仅覆盖那些仍为 None 的参数；命令行优先。"""
    for k in keys:
        if getattr(args, k, None) is None and k in cfg:
            setattr(args, k, cfg[k])

# -------------------- 主流程 --------------------
def main():
    DEF_MSG = os.path.normpath(os.path.join(ROOT, 'data', 'samples', 'msg.txt'))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "lt.json"), help="配置文件路径（JSON）")
    # 这些参数默认 None，让配置文件来填；命令行可覆盖
    ap.add_argument("--message", default=None, help="待发送的文件路径")
    ap.add_argument("--block-size", type=int, default=None, help="源块大小；None 或 0=自动选择")
    ap.add_argument("--k-min", type=int, default=None)
    ap.add_argument("--k-max", type=int, default=None)
    ap.add_argument("--allowed-sizes", default=None, help="逗号分隔或 JSON 列表")
    ap.add_argument("--base-seed", type=int, default=None)
    ap.add_argument("--loss", type=float, default=None, help="独立丢包率 p_loss ∈ [0,1)")
    ap.add_argument("--max-sent", type=int, default=None, help="最多尝试发送的包数")
    ap.add_argument("--print-every", type=int, default=None, help="每发送多少包打印一次进度")
    ap.add_argument("--out", default=None, help="成功解码后写出的文件名")
    ap.add_argument("--preview", dest="preview", action="store_true")
    ap.add_argument("--no-preview", dest="preview", action="store_false")
    ap.set_defaults(preview=None)
    args = ap.parse_args()

    # 加载配置并填充默认
    cfg = load_config(args.config, "until")
    apply_cfg(args, cfg, ["message","block_size","k_min","k_max","allowed_sizes",
                          "base_seed","loss","max_sent","print_every","out","preview"])

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
    if args.loss is None: args.loss = 0.3
    if args.max_sent is None: args.max_sent = 100000
    if args.print_every is None: args.print_every = 200
    if args.out is None: args.out = "recovered.bin"
    if args.preview is None: args.preview = False

    # 读消息
    if not os.path.isfile(args.message):
        raise FileNotFoundError(f"找不到消息文件：{args.message}")
    with open(args.message, "rb") as f:
        msg = f.read()
    msg_len = len(msg)
    if msg_len == 0:
        raise ValueError("消息为空")

    # 选 block_size
    if args.block_size and args.block_size > 0:
        bs = int(args.block_size)
        k = max(1, (msg_len + bs - 1) // bs)
        print(f"[INFO] manual block_size={bs} -> k={k} (msg_len={msg_len})")
    else:
        bs, k = choose_block_size_auto(msg_len, args.k_min, args.k_max, allowed)
        print(f"[AUTO] chose block_size={bs} (allowed={allowed}) for msg_len={msg_len} -> k={k} (target {args.k_min}..{args.k_max})")

    # 编码/解码器 + 随机数
    enc = LTEncoder(msg, block_size=bs, base_seed=args.base_seed, systematic=True)
    dec = LTDecoder()
    rng = np.random.default_rng(args.base_seed ^ 0xA5A5_5A5A)

    sent = rx = drops = 0
    t0 = time.time()

    # 直到成功或达到上限
    while not dec.is_decoded() and sent < args.max_sent:
        pkt = enc.next_packet()
        sent += 1
        if rng.random() < args.loss:
            drops += 1
            continue
        dec.add_packet(pkt)
        rx += 1
        if args.print_every > 0 and (sent % args.print_every == 0):
            print(f"[..] sent={sent}  rx={rx}  rx/k={rx/float(k):.3f}")

    elapsed = time.time() - t0
    success = dec.is_decoded()

    print("\n========== RESULT ==========")
    print(f"k={k}  block_size={bs}  msg_len={msg_len}")
    print(f"loss={args.loss:.3f}  base_seed={args.base_seed}")
    print(f"sent={sent}  received(valid)={rx}  rx/k={rx/float(k):.3f}  time={elapsed:.3f}s")
    print(f"drops (simulated loss)={drops}")

    if success:
        rec = dec.reconstruct()
        # 自动创建输出目录
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        # 写盘
        with open(args.out, "wb") as f:
            f.write(rec)
        ok = (rec == msg)
        print(f"[OK] Recovered payload. Bytes match: {ok}.")
        print(f"src_sha256={sha256_hex(msg)}")
        print(f"rec_sha256={sha256_hex(rec)}")
        print(f"→ 写出: {args.out}")

        # 预览（绝对路径、with、try/except）
        if args.preview:
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
                base = os.path.splitext(args.out)[0]
                def _save_and_open(ext: str):
                    path = os.path.abspath(base + ext)
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
                    print("[preview] 文本前 800 字符：\n" + head)
                else:
                    print("[preview] 未识别类型，前 64 字节十六进制：")
                    print(rec[:64].hex(" "))
            except Exception as e:
                print(f"[preview] 预览流程失败：{e}")
    else:
        print("[FAIL] 未能在 max-sent 内解码成功。建议降低丢包率或增大 max-sent。")

if __name__ == "__main__":
    main()
