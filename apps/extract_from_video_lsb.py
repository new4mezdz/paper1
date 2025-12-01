import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import cv2
import numpy as np
from fountain.old import LTDecoder, LTPacket
from stego.stego_lsb import extract_bits_lsb
import pickle


def extract_copyright_from_video(
        stego_i_frames_dir: str,
        max_packets: int = None
):
    """
    从隐写I帧中用LSB提取并解码版权信息
    """
    # 初始化LT解码器
    decoder = LTDecoder()

    # 获取所有隐写I帧
    i_frame_files = sorted([f for f in os.listdir(stego_i_frames_dir) if f.startswith("I_pts_")])
    print(f"找到 {len(i_frame_files)} 个隐写I帧")

    if max_packets:
        i_frame_files = i_frame_files[:max_packets]
        print(f"只处理前 {max_packets} 个I帧\n")

    packets_decoded = 0

    for i, fname in enumerate(i_frame_files):
        print(f"[{i + 1}/{len(i_frame_files)}] {fname}")

        # 读取隐写I帧
        img_path = os.path.join(stego_i_frames_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  ✗ 无法读取, 跳过\n")
            continue

        try:
            # 先提取一个合理的最大长度
            # LT包结构大约: 头部(k, block_size, msg_len等) + payload(block_size字节)
            max_pkt_bytes = 2048  # 根据你的block_size调整
            max_pkt_bits = max_pkt_bytes * 8

            # 从LSB提取比特
            extracted_bits = extract_bits_lsb(img, max_pkt_bits)

            # 转回字节
            extracted_bytes = np.packbits(extracted_bits).tobytes()

            # 尝试反序列化LT包(pickle会在遇到非法数据时抛出异常)
            pkt = pickle.loads(extracted_bytes)

            if not isinstance(pkt, LTPacket):
                print(f"  ✗ 提取的数据不是LT包\n")
                continue

            # 添加到解码器
            decoder.add_packet(pkt)
            packets_decoded += 1

            if pkt.sys_idx is not None:
                print(f"  ✓ 系统包 #{pkt.sys_idx}")
            else:
                print(f"  ✓ 冗余包 (seed={pkt.seed})")

            # 检查是否已解码完成
            if decoder.is_decoded():
                print(f"\n{'=' * 50}")
                print(f"✓ 解码成功! 使用了 {packets_decoded} 个包")
                copyright_bytes = decoder.reconstruct()
                copyright_text = copyright_bytes.decode('utf-8', errors='ignore')
                print(f"\n版权信息:\n{copyright_text}")
                print(f"{'=' * 50}\n")
                return copyright_text
            else:
                print()

        except Exception as e:
            print(f"  ✗ 提取失败: {e}\n")
            continue

    print(f"共处理 {packets_decoded} 个包")
    if decoder.is_decoded():
        print("✓ 解码成功!")
        copyright_bytes = decoder.reconstruct()
        copyright_text = copyright_bytes.decode('utf-8', errors='ignore')
        print(f"\n版权信息:\n{copyright_text}")
        return copyright_text
    else:
        print(f"✗ 解码失败: 收到 {packets_decoded} 个包, 还不足以解码")
        print(f"   (需要至少 k={decoder.k} 个包才能解码)")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从隐写I帧中提取版权信息(LSB)")
    parser.add_argument("--i-frames", required=True, help="隐写I帧文件夹")
    parser.add_argument("--max-packets", type=int, help="最多处理多少个包")

    args = parser.parse_args()
    extract_copyright_from_video(args.i_frames, args.max_packets)