# -*- coding: utf-8 -*-
"""
视频质量评估工具
对比原始视频和水印视频的 PSNR、SSIM
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ================= 配置区域 =================
ORIGINAL_VIDEO = r"F:\python\paper data\1.mp4"
WATERMARKED_VIDEO = r"F:\python\paper data\watermarked_video_lossless.mp4"
# ============================================


def calculate_psnr(img1, img2):
    """计算 PSNR"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def calculate_ssim(img1, img2):
    """计算 SSIM"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2)


def compare_videos(original_path, watermarked_path):
    """逐帧对比两个视频"""
    cap1 = cv2.VideoCapture(original_path)
    cap2 = cv2.VideoCapture(watermarked_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("错误：无法打开视频文件")
        return

    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    print(f"原始视频: {original_path}")
    print(f"水印视频: {watermarked_path}")
    print(f"总帧数: {total_frames}, FPS: {fps:.2f}")
    print("-" * 50)

    psnr_list = []
    ssim_list = []
    frame_idx = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        psnr_val = calculate_psnr(frame1, frame2)
        ssim_val = calculate_ssim(frame1, frame2)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  处理中... {frame_idx}/{total_frames}", end="\r")

    cap1.release()
    cap2.release()

    print(f"\n{'=' * 50}")
    print(f"📊 质量评估结果 (共 {len(psnr_list)} 帧)")
    print(f"{'=' * 50}")

    print(f"\n【PSNR (峰值信噪比)】")
    print(f"  平均值: {np.mean(psnr_list):.2f} dB")
    print(f"  最小值: {np.min(psnr_list):.2f} dB")
    print(f"  最大值: {np.max(psnr_list):.2f} dB")
    print(f"  标准差: {np.std(psnr_list):.2f} dB")

    print(f"\n【SSIM (结构相似度)】")
    print(f"  平均值: {np.mean(ssim_list):.4f}")
    print(f"  最小值: {np.min(ssim_list):.4f}")
    print(f"  最大值: {np.max(ssim_list):.4f}")
    print(f"  标准差: {np.std(ssim_list):.4f}")

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print(f"\n{'=' * 50}")
    print(f"📈 质量评级")
    print(f"{'=' * 50}")

    if avg_psnr > 40:
        psnr_grade = "优秀 ⭐⭐⭐"
    elif avg_psnr > 35:
        psnr_grade = "良好 ⭐⭐"
    elif avg_psnr > 30:
        psnr_grade = "一般 ⭐"
    else:
        psnr_grade = "较差 ❌"

    if avg_ssim > 0.95:
        ssim_grade = "优秀 ⭐⭐⭐"
    elif avg_ssim > 0.90:
        ssim_grade = "良好 ⭐⭐"
    elif avg_ssim > 0.80:
        ssim_grade = "一般 ⭐"
    else:
        ssim_grade = "较差 ❌"

    print(f"  PSNR 评级: {psnr_grade}")
    print(f"  SSIM 评级: {ssim_grade}")


if __name__ == "__main__":
    compare_videos(ORIGINAL_VIDEO, WATERMARKED_VIDEO)