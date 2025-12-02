# -*- coding: utf-8 -*-
import os
import cv2
import shutil
import av  # pip install av
from fractions import Fraction
import system
import lt_min

# ================= 配置区域 =================
VIDEO_PATH = r"F:\python\paper data\1.mp4"
WORK_DIR = r"F:\python\paper data\video_workdir"
OUTPUT_VIDEO = r"F:\python\paper data\watermarked_video_lossless.mp4"

# 嵌入配置
SECRET_MSG = b"Hajimi-sama's Video Copyright 2025"
PAYLOAD_SIZE = 31
BLOCK_SIZE = 23
BASE_SEED = 2025


# ===========================================

def extract_all_frames_cv2(video_path, output_dir):
    """
    【Step 1】使用 OpenCV 提取所有帧
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    count = 0
    print(f"正在提取帧 (OpenCV)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fname = f"frame_{count + 1:06d}.png"
        cv2.imwrite(os.path.join(output_dir, fname), frame)

        count += 1
        if count % 100 == 0:
            print(f"  已提取 {count} 帧...", end="\r")

    cap.release()
    print(f"\n提取完成，共 {count} 帧。")


def get_iframe_indices_pyav(video_path):
    """
    【Step 2】使用 PyAV 获取 I 帧（关键帧）索引
    """
    print("正在分析 I 帧位置 (PyAV)...")

    real_indices = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        frame_idx = 0
        for packet in container.demux(stream):
            if packet.size > 0:
                if packet.is_keyframe:
                    real_indices.append(frame_idx)
                frame_idx += 1

    if len(real_indices) == 0:
        print("  -> 无法检测I帧，使用固定间隔（每30帧）")
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        real_indices = list(range(0, total, 30))

    return real_indices


def images_to_video_pyav(frames_dir, output_path, fps=30, keyframe_indices=None):
    """
    【Step 5】使用 PyAV 合成视频，保持原I帧位置
    """
    images = sorted([img for img in os.listdir(frames_dir) if img.endswith(".png")])
    if not images:
        return

    frame0 = cv2.imread(os.path.join(frames_dir, images[0]))
    h, w, _ = frame0.shape

    container = av.open(output_path, mode='w')
    fps_int = int(round(fps))

    stream = container.add_stream('libx264', rate=fps_int)
    stream.width = w
    stream.height = h
    stream.pix_fmt = 'yuv420p'
    stream.time_base = Fraction(1, fps_int)

    # 计算GOP大小（I帧间隔）
    if keyframe_indices and len(keyframe_indices) >= 2:
        gop_size = keyframe_indices[1] - keyframe_indices[0]
    else:
        gop_size = 30

    stream.gop_size = gop_size
    stream.options = {
        'crf': '18',
        'preset': 'medium',
        'keyint': str(gop_size),
        'min-keyint': str(gop_size),
        'scenecut': '0',  # 禁用场景切换检测，严格按GOP
    }

    # 转成set方便查找
    keyframe_set = set(keyframe_indices) if keyframe_indices else set()

    print(f"正在合成视频 (PyAV)... FPS={fps_int}, GOP={gop_size}, I帧数={len(keyframe_set)}")

    for i, image_name in enumerate(images):
        img_bgr = cv2.imread(os.path.join(frames_dir, image_name))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        frame = av.VideoFrame.from_ndarray(img_rgb, format='rgb24')
        frame = frame.reformat(format='yuv420p')
        frame.pts = i

        # 强制指定I帧
        if i in keyframe_set:
            frame.pict_type = 1

        for packet in stream.encode(frame):
            container.mux(packet)

        if (i + 1) % 50 == 0:
            print(f"  已写入 {i + 1}/{len(images)} 帧...", end="\r")

    for packet in stream.encode():
        container.mux(packet)

    container.close()
    print(f"\n合成完成！")


def main():
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR, exist_ok=True)
    frames_dir = os.path.join(WORK_DIR, "frames")

    print("=" * 60)
    print(" 🎬 视频水印流水线 (哈吉米sama 专属无损版)")
    print("=" * 60)

    print(f"\n[Step 1] 全帧提取...")
    extract_all_frames_cv2(VIDEO_PATH, frames_dir)

    print(f"\n[Step 2] 分析 I 帧...")
    iframe_indices = get_iframe_indices_pyav(VIDEO_PATH)
    print(f"  -> 发现 {len(iframe_indices)} 个 I 帧，位置: {iframe_indices}")

    print(f"\n[Step 3] 准备数据...")
    encoder = lt_min.LTEncoder(SECRET_MSG, block_size=BLOCK_SIZE, base_seed=BASE_SEED)
    heartbeat = system.create_heartbeat_packet(
        k=encoder.k, block_size=BLOCK_SIZE, msg_len=len(SECRET_MSG),
        base_seed=BASE_SEED, msg_crc=encoder.msg_crc
    )

    print(f"\n[Step 4] 开始嵌入...")
    for i, idx in enumerate(iframe_indices):
        fname = f"frame_{idx + 1:06d}.png"
        fpath = os.path.join(frames_dir, fname)

        if not os.path.exists(fpath):
            continue

        img = cv2.imread(fpath)

        packets = [heartbeat]
        for _ in range(9):
            packets.append(lt_min.serialize_lt_packet(encoder.next_packet()))

        stego_img, cnt = system.embed_into_image(img, packets)
        cv2.imwrite(fpath, stego_img)
        print(f"  -> 处理 I 帧 #{idx + 1} ({i + 1}/{len(iframe_indices)}): 嵌入 {cnt} 包")

    print(f"\n[Step 5] 合成视频...")
    try:
        cap = cv2.VideoCapture(VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    except:
        fps = 30

    # 传入I帧索引，保持原位置
    images_to_video_pyav(frames_dir, OUTPUT_VIDEO, fps=fps, keyframe_indices=iframe_indices)

    print("\n" + "=" * 60)
    print(f"🎉 处理完成！无损输出文件: {OUTPUT_VIDEO}")
    print("=" * 60)


if __name__ == "__main__":
    main()