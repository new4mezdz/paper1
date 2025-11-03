# -*- coding: utf-8 -*-
"""
stego/stego_lsb.py
简单的 LSB 隐写实现
"""
import numpy as np
import cv2


def embed_bits_lsb(img_bgr: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """
    将比特序列嵌入到图像的 LSB 中
    按行优先顺序嵌入到 B, G, R 通道
    """
    img = img_bgr.copy()
    H, W, C = img.shape
    max_bits = H * W * C

    if len(bits) > max_bits:
        raise ValueError(f"比特数 {len(bits)} 超过图像容量 {max_bits}")

    # 展平图像
    flat = img.reshape(-1)

    # 嵌入比特到 LSB
    for i, bit in enumerate(bits):
        flat[i] = (flat[i] & 0xFE) | int(bit)  # 清除LSB，设置新比特

    return flat.reshape(H, W, C)


def extract_bits_lsb(img_bgr: np.ndarray, num_bits: int) -> np.ndarray:
    """
    从图像 LSB 中提取比特序列
    """
    H, W, C = img_bgr.shape
    max_bits = H * W * C

    if num_bits > max_bits:
        raise ValueError(f"请求比特数 {num_bits} 超过图像容量 {max_bits}")

    # 展平图像
    flat = img_bgr.reshape(-1)

    # 提取 LSB
    bits = np.zeros(num_bits, dtype=np.uint8)
    for i in range(num_bits):
        bits[i] = flat[i] & 1

    return bits


def compute_capacity(img_bgr: np.ndarray) -> int:
    """计算图像的 LSB 容量(比特数)"""
    H, W, C = img_bgr.shape
    return H * W * C