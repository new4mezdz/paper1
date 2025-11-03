import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import cv2
import numpy as np
from fountain.lt_min import LTEncoder, LTPacket
from fountain.auto_blocksize import resolve_block_size
from stego.stego_lsb import embed_bits_lsb, compute_capacity
import pickle
from PIL import Image

# ... 后面的代码不变

def embed_copyright_to_video(
        i_frames_dir: str,
        copyright_file: str,
        output_dir: str,
        block_size: str = "auto",  # 支持 "auto" 或具体数字
        target_k: int = 50,  # 目标包数量
        overhead: int = 100  # 序列化开销估计(字节)
):
    """
    将版权信息用LT编码后用LSB嵌入到I帧
    """
    # 1. 读取版权信息
    with open(copyright_file, 'r', encoding='utf-8') as f:
        copyright_text = f.read().strip()

    copyright_bytes = copyright_text.encode('utf-8')
    msg_len = len(copyright_bytes)

    print(f"版权信息文件: {copyright_file}")
    print(f"版权信息长度: {msg_len} 字节")
    print(f"版权信息预览: {copyright_text[:100]}{'...' if len(copyright_text) > 100 else ''}\n")

    # 2. 获取所有I帧并检查第一帧容量
    i_frame_files = sorted([f for f in os.listdir(i_frames_dir) if f.startswith("I_pts_")])
    print(f"找到 {len(i_frame_files)} 个I帧")

    if len(i_frame_files) == 0:
        print("错误: 没有找到I帧!")
        return

    # 读取第一帧获取容量信息
    first_frame_path = os.path.join(i_frames_dir, i_frame_files[0])
    first_img = cv2.imread(first_frame_path, cv2.IMREAD_COLOR)
    if first_img is None:
        print("错误: 无法读取第一帧!")
        return

    # 计算单帧LSB容量(字节)
    frame_capacity_bits = compute_capacity(first_img)
    frame_capacity_bytes = frame_capacity_bits // 8

    print(f"单帧容量: {frame_capacity_bytes} 字节 ({frame_capacity_bits} 比特)")
    print(f"图像尺寸: {first_img.shape[1]}x{first_img.shape[0]}\n")

    # 3. 使用 auto_blocksize 计算合适的 block_size
    # 注意: 需要考虑 pickle 序列化后的额外开销
    # LTPacket 序列化后大约是: block_size + overhead
    max_payload_bytes = frame_capacity_bytes - overhead

    print("=" * 60)
    print("自动计算 block_size...")
    print("=" * 60)

    result = resolve_block_size(
        arg_block_size=block_size,
        msg_len=msg_len,
        cover_img=Image.fromarray(cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)),
        target_k=target_k,
        cap_bytes_override=max_payload_bytes,  # 使用实际LSB容量
        min_bs=64,
        align=16
    )

    chosen_bs = result.chosen_block_size
    k_estimate = result.k_estimate

    print(f"选择的 block_size: {chosen_bs} 字节")
    print(f"预计包数量 k: {k_estimate}")
    print(f"选择原因: {result.reason}")
    if result.clipped:
        print(f"⚠ 警告: {result.advice}")
    print(f"容量来源: {result.cap_source}")
    print("=" * 60 + "\n")

    # 检查是否有足够的I帧
    if k_estimate > len(i_frame_files):
        print(f"⚠ 警告: 需要 {k_estimate} 个I帧, 但只有 {len(i_frame_files)} 个")
        print(f"建议: 使用更长的视频或减小 target_k\n")

    # 4. LT编码
    encoder = LTEncoder(copyright_bytes, block_size=chosen_bs)
    print(f"LT编码器初始化完成: k={encoder.k}, block_size={chosen_bs}\n")

    # 5. 为每个I帧生成并嵌入LT包
    os.makedirs(output_dir, exist_ok=True)

    success_count = 0

    for i, fname in enumerate(i_frame_files):
        # 生成LT包
        pkt = encoder.next_packet()

        # 序列化包
        pkt_bytes = pickle.dumps(pkt)
        pkt_bits = np.unpackbits(np.frombuffer(pkt_bytes, dtype=np.uint8))

        print(f"[{i + 1}/{len(i_frame_files)}] {fname}")
        if pkt.sys_idx is not None:
            print(f"  包类型: 系统包 #{pkt.sys_idx}")
        else:
            print(f"  包类型: 冗余包 (seed={pkt.seed})")
        print(f"  包大小: {len(pkt_bytes)} 字节 = {len(pkt_bits)} 比特")

        # 读取I帧
        img_path = os.path.join(i_frames_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  ✗ 无法读取, 跳过\n")
            continue

        # 检查容量
        capacity = compute_capacity(img)

        if len(pkt_bits) > capacity:
            print(f"  ✗ 容量不足! 需要 {len(pkt_bits)} 比特, 只有 {capacity} 比特")
            print(f"  这不应该发生,请检查配置\n")
            continue

        # LSB嵌入
        try:
            stego_img = embed_bits_lsb(img, pkt_bits)

            # 保存
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, stego_img)
            success_count += 1
            print(f"  ✓ 已保存到: {out_path}\n")
        except Exception as e:
            print(f"  ✗ 嵌入失败: {e}\n")

    print("=" * 60)
    print(f"完成! 成功嵌入 {success_count}/{len(i_frame_files)} 个I帧")
    print(f"隐写I帧保存在: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将版权信息用LT码+LSB嵌入视频I帧(自动block_size)")
    parser.add_argument("--i-frames", required=True, help="提取的I帧文件夹")
    parser.add_argument("--copyright-file", required=True, help="版权信息文本文件路径")
    parser.add_argument("--output", required=True, help="输出隐写I帧文件夹")
    parser.add_argument("--block-size", default="auto", help="LT块大小: 'auto' 或具体数字(字节)")
    parser.add_argument("--target-k", type=int, default=50, help="目标包数量(用于auto模式)")
    parser.add_argument("--overhead", type=int, default=100, help="序列化开销估计(字节)")

    args = parser.parse_args()
    embed_copyright_to_video(
        args.i_frames,
        args.copyright_file,
        args.output,
        args.block_size,
        args.target_k,
        args.overhead
    )