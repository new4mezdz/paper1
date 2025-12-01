# -*- coding: utf-8 -*-
import os
import json
import cv2
import shutil
import av  # 只要 pip install av 就能用，不需要配环境变量
import system
import lt_min

# ================= 配置区域 =================
VIDEO_PATH = r"D:\paper data\3.mp4"  # 您的输入视频
WORK_DIR = r"D:\paper data\video_workdir"
OUTPUT_VIDEO = r"D:\paper data\watermarked_video.mp4"

# 嵌入配置
SECRET_MSG = b"Hajimi-sama's Video Copyright 2025"
PAYLOAD_SIZE = 31
BLOCK_SIZE = 23
BASE_SEED = 2025


# ===========================================

def extract_all_frames_cv2(video_path, output_dir):
    """
    【替代 FFmpeg】使用 OpenCV 提取所有帧
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

        # 保存为 png 无损
        fname = f"frame_{count + 1:06d}.png"
        cv2.imwrite(os.path.join(output_dir, fname), frame)

        count += 1
        if count % 100 == 0:
            print(f"  已提取 {count} 帧...", end="\r")

    cap.release()
    print(f"\n提取完成，共 {count} 帧。")


def get_iframe_indices_pyav(video_path):
    """
    【替代 ffprobe】使用 PyAV 获取 I 帧索引
    """
    indices = []
    print("正在分析 I 帧位置 (PyAV)...")

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        # 只需要遍历包，不需要解码图像，速度很快
        for packet in container.demux(stream):
            if packet.dts is None:
                continue

            # 只有关键帧才记录
            if packet.is_keyframe:
                # 这种方法获取的是大概的帧序，通常足够准确
                # 如果需要绝对精确，可能需要 decode，但速度慢
                # 这里为了速度，我们假设 I 帧就是 Keyframe
                # PyAV 这里的逻辑可能需要根据具体视频微调，但在 MP4 里通常是对的
                pass

    # 为了绝对精确，我们还是解码一遍吧（反正只用跑一次）
    # 重新打开以进行解码扫描
    real_indices = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if frame.pict_type == 'I':
                real_indices.append(i)

    return real_indices


def images_to_video_cv2(frames_dir, output_path, fps=30):
    """
    【替代 FFmpeg】使用 OpenCV 合成视频
    """
    images = sorted([img for img in os.listdir(frames_dir) if img.endswith(".png")])
    if not images:
        return

    frame0 = cv2.imread(os.path.join(frames_dir, images[0]))
    h, w, layers = frame0.shape

    # 'mp4v' 是最通用的编码，不需要额外安装
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"正在合成视频 (OpenCV)... FPS={fps}")
    for i, image in enumerate(images):
        frame = cv2.imread(os.path.join(frames_dir, image))
        out.write(frame)
        if i % 100 == 0:
            print(f"  已写入 {i} 帧...", end="\r")

    out.release()
    print("\n合成完成！")


def main():
    # 0. 准备环境
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR, exist_ok=True)
    frames_dir = os.path.join(WORK_DIR, "frames")

    print("=" * 60)
    print(" 🎬 视频水印流水线 (纯 Python 版)")
    print("=" * 60)

    # 1. 提取所有帧
    print(f"\n[Step 1] 全帧提取...")
    extract_all_frames_cv2(VIDEO_PATH, frames_dir)

    # 2. 识别 I 帧
    print(f"\n[Step 2] 分析 I 帧...")
    iframe_indices = get_iframe_indices_pyav(VIDEO_PATH)
    print(f"  -> 发现 {len(iframe_indices)} 个 I 帧: {iframe_indices[:10]}...")

    # 3. 准备数据
    print(f"\n[Step 3] 准备数据...")
    encoder = lt_min.LTEncoder(SECRET_MSG, block_size=BLOCK_SIZE, base_seed=BASE_SEED)
    heartbeat = system.create_heartbeat_packet(
        k=encoder.k, block_size=BLOCK_SIZE, msg_len=len(SECRET_MSG),
        base_seed=BASE_SEED, msg_crc=encoder.msg_crc
    )

    # 4. 定向嵌入
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

    # 5. 还原视频
    print(f"\n[Step 5] 合成视频...")
    # 获取原视频帧率
    try:
        cap = cv2.VideoCapture(VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    except:
        fps = 30  # 默认

    images_to_video_cv2(frames_dir, OUTPUT_VIDEO, fps=fps)

    print("\n" + "=" * 60)
    print(f"🎉 处理完成！输出文件: {OUTPUT_VIDEO}")
    print("=" * 60)


if __name__ == "__main__":
    main()