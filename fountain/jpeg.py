# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim

# =================配置区域=================
# 请将此路径修改为您本地的图片路径
IMG_PATH = r"D:\paper data\output_3\I\I_pts_90090.png"
# 测试的压缩质量列表 (100=最好, 1=最差)
QUALITIES = [100, 80, 60, 40, 20, 10]


# ==========================================

def calculate_metrics(original, compressed):
    """计算 PSNR 和 SSIM"""
    # 1. 计算 PSNR
    psnr = cv2.PSNR(original, compressed)

    # 2. 计算 SSIM
    # 为了兼容性，先转为灰度图计算 (SSIM在彩色图上计算较慢且复杂)
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_comp = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    s = ssim(gray_orig, gray_comp)

    return psnr, s


def main():
    # 1. 读取图片
    if not os.path.exists(IMG_PATH):
        print(f"错误: 找不到图片 {IMG_PATH}")
        return

    img_orig = cv2.imread(IMG_PATH)
    print(f"成功读取图片: {IMG_PATH} | 分辨率: {img_orig.shape}")

    # 准备数据记录
    psnr_scores = []
    ssim_scores = []
    file_sizes = []
    compressed_imgs = []

    print("\n" + "=" * 60)
    print(f"{'Quality':<8} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Size (KB)':<10}")
    print("-" * 60)

    # 2. 循环测试不同质量
    for q in QUALITIES:
        # 模拟压缩: 编码 -> 解码
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        # 编码 (得到压缩后的二进制数据)
        result, encimg = cv2.imencode('.jpg', img_orig, encode_param)
        # 解码 (得到压缩后的图像矩阵)
        img_comp = cv2.imdecode(encimg, 1)

        # 计算文件大小 (KB)
        size_kb = len(encimg) / 1024
        file_sizes.append(size_kb)

        # 计算指标
        psnr_val, ssim_val = calculate_metrics(img_orig, img_comp)
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)

        # 保存图像以便稍后画图 (转RGB)
        img_rgb = cv2.cvtColor(img_comp, cv2.COLOR_BGR2RGB)
        compressed_imgs.append(img_rgb)

        print(f"{q:<8} | {psnr_val:<10.2f} | {ssim_val:<8.4f} | {size_kb:<10.2f}")

    print("=" * 60)

    # 3. 可视化：生成对比大图
    print("\n正在生成可视化图表...")
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"JPEG Compression Analysis: {os.path.basename(IMG_PATH)}", fontsize=16)

    for i, q in enumerate(QUALITIES):
        plt.subplot(2, 3, i + 1)
        plt.imshow(compressed_imgs[i])
        plt.title(f"Quality={q}\nPSNR={psnr_scores[i]:.2f}dB | SSIM={ssim_scores[i]:.3f}\nSize={file_sizes[i]:.1f}KB")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('jpeg_visual_comparison.png')  # 保存对比图
    print("  -> 已保存图片对比图: jpeg_visual_comparison.png")

    # 4. 可视化：生成数据曲线图
    plt.figure(figsize=(12, 5))

    # PSNR 曲线
    plt.subplot(1, 2, 1)
    plt.plot(QUALITIES, psnr_scores, 'bo-', linewidth=2)
    plt.title('Quality vs PSNR (Higher is Better)')
    plt.xlabel('JPEG Quality')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    plt.gca().invert_xaxis()  # X轴反转，让大数字(高质量)在左边，或者保留默认

    # SSIM 曲线
    plt.subplot(1, 2, 2)
    plt.plot(QUALITIES, ssim_scores, 'ro-', linewidth=2)
    plt.title('Quality vs SSIM (Higher is Better)')
    plt.xlabel('JPEG Quality')
    plt.ylabel('SSIM')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('jpeg_metrics_curve.png')  # 保存曲线图
    print("  -> 已保存数据曲线图: jpeg_metrics_curve.png")

    plt.show()  # 弹窗显示


if __name__ == "__main__":
    main()