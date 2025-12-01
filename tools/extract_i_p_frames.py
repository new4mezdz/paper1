import os
import av
import argparse
from typing import Tuple

"""
把原脚本扩展为两种模式：
  1) 只输出 I 帧（--mode i）
  2) 输出 I 帧 和 P 帧（--mode ip）

依赖：PyAV（pip install av）
示例：
  python extract_i_p_frames.py --video "D:/paper/data/samples/1.mp4" --out extracted_frames --mode ip
  python extract_i_p_frames.py --video "D:/paper/data/samples/2.mp4" --out extracted_i_only --mode i
"""

PICT_I = "I"
PICT_P = "P"


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def pict_type_of(frame) -> str:
    """安全获取帧类型：'I' / 'P' / 'B' / None"""
    try:
        # PyAV: frame.pict_type 是枚举，str() 后一般为 'I'/'P'/'B'
        pt = getattr(frame, "pict_type", None)
        if pt is None:
            return None
        # 一些环境里 pt 可能是枚举对象，.name/.value 都可能可用
        if hasattr(pt, "name"):
            return pt.name
        return str(pt)
    except Exception:
        return None


def should_save(frame, mode: str) -> Tuple[bool, str]:
    """
    根据模式判断当前帧是否保存。
    返回: (是否保存, 标签 'I' 或 'P')；若不保存则 (False, '')。
    """
    pt = pict_type_of(frame)

    if mode == "i":
        # 只保存 I 帧：优先用 pict_type 判断；回退到 key_frame
        if pt == PICT_I or frame.key_frame:
            return True, PICT_I
        return False, ""

    if mode == "ip":
        # 保存 I 或 P 帧
        if pt == PICT_I or frame.key_frame:
            return True, PICT_I
        if pt == PICT_P:
            return True, PICT_P
        return False, ""

    # 未知模式
    return False, ""


def extract_frames(video_path: str, output_dir: str, mode: str = "i") -> None:
    # 校验
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在：{video_path}")
        return

    mode = mode.lower().strip()
    if mode not in {"i", "ip"}:
        print("错误：--mode 只支持 'i' 或 'ip'")
        return

    # 目录结构：
    #   output_dir/
    #       I/  ...
    #       P/  ...  (仅在 mode=ip 时创建)
    ensure_dir(output_dir)
    out_I = os.path.join(output_dir, "I")
    out_P = os.path.join(output_dir, "P")
    ensure_dir(out_I)
    if mode == "ip":
        ensure_dir(out_P)

    saved_I = 0
    saved_P = 0

    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            print(f"开始解析: {os.path.basename(video_path)} | mode={mode} | codec={stream.codec.name} | fps≈{float(stream.average_rate) if stream.average_rate else '未知'}")

            for frame in container.decode(stream):
                save, label = should_save(frame, mode)
                if not save:
                    continue

                img = frame.to_image()
                ts = None
                try:
                    ts = f"{frame.time:.3f}s" if frame.time is not None else "n/a"
                except Exception:
                    ts = "n/a"

                pts = frame.pts if frame.pts is not None else 0

                # ==========================================
                # 修改点：后缀名改为 .png 以保证无损
                # ==========================================
                if label == PICT_I:
                    out_dir = out_I
                    fname = os.path.join(out_dir, f"I_pts_{pts}.png") # <--- 改为 .png
                    img.save(fname, "PNG")
                    saved_I += 1
                    print(f"保存 I 帧: pts={pts}, t={ts} -> {fname}")
                elif label == PICT_P:
                    out_dir = out_P
                    fname = os.path.join(out_dir, f"P_pts_{pts}.png") # <--- 改为 .png
                    img.save(fname, "PNG")
                    saved_P += 1
                    print(f"保存 P 帧: pts={pts}, t={ts} -> {fname}")

        # 结果汇总
        if mode == "i":
            if saved_I == 0:
                print("完成：未发现任何 I 帧。")
            else:
                print(f"完成：共导出 I 帧 {saved_I} 张。输出文件夹：{os.path.abspath(out_I)}")
        else:
            print(f"完成：I 帧 {saved_I} 张，P 帧 {saved_P} 张。")
            print(f"I -> {os.path.abspath(out_I)}")
            print(f"P -> {os.path.abspath(out_P)}")

    except av.AVError as e:
        print(f"处理视频时发生 AV 错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="按模式导出视频的 I 帧 或 I+P 帧 (PyAV)"
    )
    ap.add_argument("--video", required=True, help="输入视频路径，例如 D:/video.mp4")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--mode", choices=["i", "ip"], default="i", help="i=仅 I 帧；ip=I+P 帧")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_frames(args.video, args.out, args.mode)