#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_images_lsb.py
从隐写图片集中提取版权信息（带魔数和CRC校验）
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import cv2
import numpy as np
import pickle
import zlib

from fountain.old import LTDecoder, LTPacket

# 协议常量（必须与嵌入端一致）
MAGIC = b'LTPK'
HEADER_SIZE = 12


def extract_bits_lsb(img_bgr: np.ndarray, num_bits: int) -> np.ndarray:
    """从图像LSB中提取比特序列"""
    H, W, C = img_bgr.shape
    max_bits = H * W * C

    if num_bits > max_bits:
        raise ValueError(f"请求比特数 {num_bits} 超过图像容量 {max_bits}")

    flat = img_bgr.reshape(-1)
    bits = np.zeros(num_bits, dtype=np.uint8)
    for i in range(num_bits):
        bits[i] = flat[i] & 1

    return bits


def extract_packet_from_image(img_bgr: np.ndarray) -> tuple:
    """
    从图像中提取一个LT包（带协议头验证）
    返回: (成功?, LTPacket或None, 错误信息)
    """
    try:
        # 1. 读取魔数 (4字节 = 32比特)
        magic_bits = extract_bits_lsb(img_bgr, 32)
        magic_bytes = np.packbits(magic_bits).tobytes()

        if magic_bytes != MAGIC:
            return False, None, f"魔数不匹配: 期望 {MAGIC.hex()}, 实际 {magic_bytes.hex()}"

        # 2. 读取长度 (4字节 = 32比特)
        len_bits = extract_bits_lsb(img_bgr, 64)[32:64]
        pkt_len = int.from_bytes(np.packbits(len_bits).tobytes(), 'big')

        # 3. 长度合理性检查
        if pkt_len <= 0 or pkt_len > 100000:
            return False, None, f"包长度异常: {pkt_len} 字节"

        # 4. 读取CRC (4字节 = 32比特)
        crc_bits = extract_bits_lsb(img_bgr, 96)[64:96]
        expected_crc = int.from_bytes(np.packbits(crc_bits).tobytes(), 'big')

        # 5. 读取数据
        total_bits = 96 + pkt_len * 8
        all_bits = extract_bits_lsb(img_bgr, total_bits)
        data_bits = all_bits[96:]
        data_bytes = np.packbits(data_bits).tobytes()[:pkt_len]

        # 6. CRC校验
        actual_crc = zlib.crc32(data_bytes) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            return False, None, f"CRC校验失败: 期望 {expected_crc:08x}, 实际 {actual_crc:08x}"

        # 7. 反序列化LT包
        pkt = pickle.loads(data_bytes)

        if not isinstance(pkt, LTPacket):
            return False, None, f"反序列化后不是LTPacket类型: {type(pkt)}"

        return True, pkt, "成功"

    except Exception as e:
        return False, None, f"提取异常: {str(e)}"


def extract_from_images(
        images_dir: str,
        max_packets: int = None
):
    """
    从隐写图片集中提取版权信息
    """
    print("=" * 70)
    print("LT喷泉码 + LSB隐写提取系统 (带魔数和CRC校验)")
    print("=" * 70)

    # 获取所有图片文件
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])

    print(f"\n🖼️  找到 {len(image_files)} 个图片文件")

    if max_packets:
        image_files = image_files[:max_packets]
        print(f"⚙️  只处理前 {max_packets} 张图片")

    # 初始化LT解码器
    decoder = LTDecoder()

    print(f"\n{'=' * 70}")
    print("开始提取...")
    print(f"{'=' * 70}\n")

    # 统计信息
    valid_packets = 0
    magic_fail = 0
    crc_fail = 0
    other_fail = 0

    for i, fname in enumerate(image_files):
        print(f"[{i + 1}/{len(image_files)}] {fname}")

        # 读取图片
        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"  ❌ 无法读取图片\n")
            other_fail += 1
            continue

        # 提取包
        success, pkt, msg = extract_packet_from_image(img)

        if not success:
            if "魔数不匹配" in msg:
                print(f"  ⚠️  {msg}")
                magic_fail += 1
            elif "CRC校验失败" in msg:
                print(f"  ❌ {msg}")
                crc_fail += 1
            else:
                print(f"  ❌ {msg}")
                other_fail += 1
            print()
            continue

        # 成功提取
        valid_packets += 1

        if pkt.sys_idx is not None:
            print(f"  ✅ 系统包 #{pkt.sys_idx}")
        else:
            print(f"  ✅ 冗余包 (seed={pkt.seed})")

        print(f"  📏 包大小: {pkt.block_size} 字节")

        # 添加到解码器
        decoder.add_packet(pkt)

        # 检查是否已解码完成
        if decoder.is_decoded():
            print(f"\n{'=' * 70}")
            print(f"🎉 解码成功! 使用了 {valid_packets} 个有效包")
            print(f"{'=' * 70}\n")

            copyright_bytes = decoder.reconstruct()
            copyright_text = copyright_bytes.decode('utf-8', errors='ignore')

            print(f"📄 版权信息:")
            print(f"{'-' * 70}")
            print(copyright_text)
            print(f"{'-' * 70}\n")

            # 统计信息
            print(f"📊 提取统计:")
            print(f"  - 有效包: {valid_packets}")
            print(f"  - 魔数错误: {magic_fail}")
            print(f"  - CRC失败: {crc_fail}")
            print(f"  - 其他错误: {other_fail}")
            print(f"  - 总处理: {i + 1}/{len(image_files)}")

            return copyright_text
        else:
            # 显示解码进度
            if decoder.initialized:
                decoded_blocks = np.sum(decoder.known_mask)
                print(f"  📊 解码进度: {decoded_blocks}/{decoder.k} 块")
            print()

    # 处理完所有图片
    print(f"\n{'=' * 70}")
    print(f"处理完成")
    print(f"{'=' * 70}\n")

    print(f"📊 提取统计:")
    print(f"  - 有效包: {valid_packets}")
    print(f"  - 魔数错误: {magic_fail}")
    print(f"  - CRC失败: {crc_fail}")
    print(f"  - 其他错误: {other_fail}")
    print(f"  - 总处理: {len(image_files)}")

    if decoder.is_decoded():
        print(f"\n✅ 解码成功!")
        copyright_bytes = decoder.reconstruct()
        copyright_text = copyright_bytes.decode('utf-8', errors='ignore')

        print(f"\n📄 版权信息:")
        print(f"{'-' * 70}")
        print(copyright_text)
        print(f"{'-' * 70}\n")

        return copyright_text
    else:
        print(f"\n❌ 解码失败: 有效包数量不足")
        if decoder.initialized:
            decoded_blocks = np.sum(decoder.known_mask)
            print(f"  当前进度: {decoded_blocks}/{decoder.k} 块")
            print(f"  还需要: 至少 {decoder.k - decoded_blocks} 个有效包")
        else:
            print(f"  未收到任何有效的LT包")

        return None


def main():
    parser = argparse.ArgumentParser(
        description="从隐写图片集中提取版权信息(带魔数和CRC校验)"
    )
    parser.add_argument(
        "--images",
        required=True,
        help="隐写图片文件夹路径"
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        help="最多处理多少张图片(用于测试)"
    )

    args = parser.parse_args()

    extract_from_images(args.images, args.max_packets)


if __name__ == "__main__":
    main()