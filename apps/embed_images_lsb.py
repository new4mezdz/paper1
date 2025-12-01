#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
embed_images_lsb.py
将版权信息用LT码+LSB嵌入到图片集中（带魔数和CRC校验）
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import cv2
import numpy as np
import pickle
import zlib
from PIL import Image

from fountain.old import LTEncoder, LTPacket
from fountain.auto_blocksize import resolve_block_size

# 协议常量
MAGIC = b'LTPK'  # 魔数：4字节
HEADER_SIZE = 12  # 魔数(4) + 长度(4) + CRC(4) = 12字节


def compute_lsb_capacity(img_bgr: np.ndarray) -> int:
    """计算图像的LSB容量(比特数)"""
    H, W, C = img_bgr.shape
    return H * W * C


def embed_bits_lsb(img_bgr: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """将比特序列嵌入到图像的LSB中"""
    img = img_bgr.copy()
    H, W, C = img.shape
    max_bits = H * W * C

    if len(bits) > max_bits:
        raise ValueError(f"比特数 {len(bits)} 超过图像容量 {max_bits}")

    # 展平图像并嵌入
    flat = img.reshape(-1)
    for i, bit in enumerate(bits):
        flat[i] = (flat[i] & 0xFE) | int(bit)

    return flat.reshape(H, W, C)


def create_packet_with_header(pkt: LTPacket) -> bytes:
    """
    创建带协议头的数据包
    格式: [魔数 4字节][长度 4字节][CRC32 4字节][LTPacket数据]
    """
    # 序列化LT包
    pkt_bytes = pickle.dumps(pkt)
    pkt_len = len(pkt_bytes)

    # 计算CRC32
    crc = zlib.crc32(pkt_bytes) & 0xFFFFFFFF

    # 构造完整包
    header = MAGIC + pkt_len.to_bytes(4, 'big') + crc.to_bytes(4, 'big')
    full_packet = header + pkt_bytes

    return full_packet


def embed_to_images(
        images_dir: str,
        copyright_file: str,
        output_dir: str,
        block_size: str = "auto",
        target_k: int = 50,
        overhead: int = 150
):
    """
    将版权信息嵌入到图片集中
    """
    print("=" * 70)
    print("LT喷泉码 + LSB隐写系统 (带魔数和CRC校验)")
    print("=" * 70)

    # 1. 读取版权信息
    with open(copyright_file, 'r', encoding='utf-8') as f:
        copyright_text = f.read().strip()

    copyright_bytes = copyright_text.encode('utf-8')
    msg_len = len(copyright_bytes)

    print(f"\n📄 版权信息文件: {copyright_file}")
    print(f"📏 版权信息长度: {msg_len} 字节")
    print(f"📝 版权信息预览: {copyright_text[:80]}{'...' if len(copyright_text) > 80 else ''}")

    # 2. 获取所有图片文件
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])

    if len(image_files) == 0:
        print(f"\n❌ 错误: 在 {images_dir} 中没有找到图片文件!")
        return

    print(f"\n🖼️  找到 {len(image_files)} 个图片文件")

    # 3. 读取第一张图片获取容量信息
    first_img_path = os.path.join(images_dir, image_files[0])
    first_img = cv2.imread(first_img_path, cv2.IMREAD_COLOR)

    if first_img is None:
        print(f"\n❌ 错误: 无法读取第一张图片 {first_img_path}")
        return

    # 计算单帧LSB容量
    frame_capacity_bits = compute_lsb_capacity(first_img)
    frame_capacity_bytes = frame_capacity_bits // 8

    print(f"\n📊 图片信息:")
    print(f"  - 尺寸: {first_img.shape[1]}x{first_img.shape[0]}")
    print(f"  - 单张容量: {frame_capacity_bytes} 字节 ({frame_capacity_bits} 比特)")

    # 4. 自动计算 block_size
    # 考虑协议头开销: 魔数(4) + 长度(4) + CRC(4) + pickle开销(~overhead)
    max_payload_bytes = frame_capacity_bytes - HEADER_SIZE - overhead

    print(f"\n⚙️  自动计算 block_size...")
    print(f"  - 协议头开销: {HEADER_SIZE} 字节")
    print(f"  - 序列化开销估计: {overhead} 字节")
    print(f"  - 可用payload空间: {max_payload_bytes} 字节")

    result = resolve_block_size(
        arg_block_size=block_size,
        msg_len=msg_len,
        cover_img=Image.fromarray(cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)),
        target_k=target_k,
        cap_bytes_override=max_payload_bytes,
        min_bs=64,
        align=16,
    force_k = 4  #
    )

    chosen_bs = result.chosen_block_size
    k_estimate = result.k_estimate

    print(f"\n✅ Block Size 计算结果:")
    print(f"  - 选择的 block_size: {chosen_bs} 字节")
    print(f"  - 预计包数量 k: {k_estimate}")
    print(f"  - 选择原因: {result.reason}")

    if result.clipped:
        print(f"  ⚠️  警告: {result.advice}")

    # 检查是否有足够的图片
    if k_estimate > len(image_files):
        print(f"\n⚠️  警告: 需要 {k_estimate} 张图片, 但只有 {len(image_files)} 张")
        print(f"  建议: 增加图片数量或减小 target_k")

    # 5. 初始化LT编码器
    encoder = LTEncoder(copyright_bytes, block_size=chosen_bs)
    print(f"\n🔧 LT编码器初始化: k={encoder.k}, block_size={chosen_bs}")

    # 6. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 7. 为每张图片生成并嵌入LT包
    print(f"\n{'=' * 70}")
    print("开始嵌入...")
    print(f"{'=' * 70}\n")

    success_count = 0

    for i, fname in enumerate(image_files):
        print(f"[{i + 1}/{len(image_files)}] {fname}")

        # 生成LT包
        pkt = encoder.next_packet()

        # 创建带协议头的完整包
        full_packet = create_packet_with_header(pkt)
        full_packet_bits = np.unpackbits(np.frombuffer(full_packet, dtype=np.uint8))

        # 包信息
        if pkt.sys_idx is not None:
            print(f"  📦 包类型: 系统包 #{pkt.sys_idx}")
        else:
            print(f"  📦 包类型: 冗余包 (seed={pkt.seed})")

        pkt_bytes = pickle.dumps(pkt)
        print(f"  📏 包大小: {len(pkt_bytes)} 字节 (含12字节头共 {len(full_packet)} 字节)")
        print(f"  🔢 总比特数: {len(full_packet_bits)} 比特")

        # 读取图片
        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"  ❌ 无法读取, 跳过\n")
            continue

        # 检查容量
        capacity = compute_lsb_capacity(img)

        if len(full_packet_bits) > capacity:
            print(f"  ❌ 容量不足! 需要 {len(full_packet_bits)} 比特, 只有 {capacity} 比特")
            print(f"  建议: 减小 block_size\n")
            continue

        # LSB嵌入
        try:
            stego_img = embed_bits_lsb(img, full_packet_bits)

            # 保存
            # 保存为无损 PNG
            base_name, _ = os.path.splitext(fname)
            out_path = os.path.join(output_dir, base_name + ".png")
            cv2.imwrite(out_path, stego_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            success_count += 1

            print(f"  ✅ 已保存为PNG: {out_path}\n")


        except Exception as e:
            print(f"  ❌ 嵌入失败: {e}\n")

    # 8. 总结
    print(f"{'=' * 70}")
    print(f"✅ 完成! 成功嵌入 {success_count}/{len(image_files)} 张图片")
    print(f"{'=' * 70}")
    print(f"\n📁 隐写图片保存在: {output_dir}")
    print(f"📊 统计信息:")
    print(f"  - 版权信息: {msg_len} 字节")
    print(f"  - LT参数: k={encoder.k}, block_size={chosen_bs}")
    print(f"  - 协议: 魔数(LTPK) + 长度 + CRC32")
    print(f"  - 成功率: {success_count}/{len(image_files)} ({100 * success_count // len(image_files)}%)")
    print(f"\n提示: 使用 extract_images_lsb.py 提取版权信息")


def main():
    parser = argparse.ArgumentParser(
        description="将版权信息用LT码+LSB嵌入图片集(带魔数和CRC校验)"
    )
    parser.add_argument(
        "--images",
        required=True,
        help="输入图片文件夹路径"
    )
    parser.add_argument(
        "--copyright-file",
        required=True,
        help="版权信息文本文件路径"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出隐写图片文件夹路径"
    )
    parser.add_argument(
        "--block-size",
        default="auto",
        help="LT块大小: 'auto' 或具体数字(字节), 默认 auto"
    )
    parser.add_argument(
        "--target-k",
        type=int,
        default=50,
        help="目标包数量(用于auto模式), 默认 50"
    )
    parser.add_argument(
        "--overhead",
        type=int,
        default=150,
        help="序列化开销估计(字节), 默认 150"
    )

    args = parser.parse_args()

    embed_to_images(
        args.images,
        args.copyright_file,
        args.output,
        args.block_size,
        args.target_k,
        args.overhead
    )


if __name__ == "__main__":
    main()