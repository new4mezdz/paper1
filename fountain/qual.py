# -*- coding: utf-8 -*-
"""
图像质量评价系统
支持：PSNR, SSIM, MSE 等指标
可评价单张图片或批量评价文件夹
"""
import cv2
import numpy as np
import os
from glob import glob

try:
    from skimage.metrics import structural_similarity as ssim

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("警告: skimage 未安装，SSIM 将使用简化版本")
    print("安装命令: pip install scikit-image")


# ==========================================
# 基础指标计算
# ==========================================

def calc_mse(original, stego):
    """计算均方误差 (Mean Squared Error)"""
    original = original.astype(np.float64)
    stego = stego.astype(np.float64)
    return np.mean((original - stego) ** 2)


def calc_psnr(original, stego):
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio)
    值越大越好，一般 >40dB 优秀，>35dB 良好，>30dB 可接受
    """
    mse = calc_mse(original, stego)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def calc_ssim(original, stego):
    """
    计算结构相似性 (Structural Similarity Index)
    范围 0~1，越接近 1 越好
    """
    if SKIMAGE_AVAILABLE:
        # 如果是彩色图，转灰度
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            stego = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)
        return ssim(original, stego)
    else:
        # 简化版 SSIM
        return calc_ssim_simple(original, stego)


def calc_ssim_simple(original, stego):
    """简化版 SSIM（不依赖 skimage）"""
    if len(original.shape) == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        stego = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)

    original = original.astype(np.float64)
    stego = stego.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(original, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(stego, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(original ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(stego ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(original * stego, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


def calc_ncc(original, stego):
    """
    计算归一化互相关系数 (Normalized Cross-Correlation)
    范围 -1~1，越接近 1 越好
    """
    if len(original.shape) == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        stego = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)

    original = original.astype(np.float64).flatten()
    stego = stego.astype(np.float64).flatten()

    original = original - np.mean(original)
    stego = stego - np.mean(stego)

    numerator = np.sum(original * stego)
    denominator = np.sqrt(np.sum(original ** 2) * np.sum(stego ** 2))

    if denominator == 0:
        return 0
    return numerator / denominator


def calc_mae(original, stego):
    """计算平均绝对误差 (Mean Absolute Error)"""
    original = original.astype(np.float64)
    stego = stego.astype(np.float64)
    return np.mean(np.abs(original - stego))


def calc_snr(original, stego):
    """计算信噪比 (Signal-to-Noise Ratio)"""
    original = original.astype(np.float64)
    stego = stego.astype(np.float64)

    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - stego) ** 2)

    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


# ==========================================
# 综合评价
# ==========================================

def evaluate_image_quality(original, stego):
    """
    综合评价图像质量
    返回字典包含所有指标
    """
    results = {
        'MSE': calc_mse(original, stego),
        'PSNR': calc_psnr(original, stego),
        'SSIM': calc_ssim(original, stego),
        'NCC': calc_ncc(original, stego),
        'MAE': calc_mae(original, stego),
        'SNR': calc_snr(original, stego),
    }
    return results


def get_quality_grade(psnr, ssim_val):
    """根据 PSNR 和 SSIM 给出等级评价"""
    if psnr > 45 and ssim_val > 0.98:
        return "优秀 (Excellent)"
    elif psnr > 40 and ssim_val > 0.95:
        return "很好 (Very Good)"
    elif psnr > 35 and ssim_val > 0.90:
        return "良好 (Good)"
    elif psnr > 30 and ssim_val > 0.85:
        return "可接受 (Acceptable)"
    else:
        return "较差 (Poor)"


def print_quality_report(original, stego, title=""):
    """打印详细质量报告"""
    results = evaluate_image_quality(original, stego)
    grade = get_quality_grade(results['PSNR'], results['SSIM'])

    print("\n" + "=" * 50)
    if title:
        print(f"  图像质量评价报告: {title}")
    else:
        print("  图像质量评价报告")
    print("=" * 50)
    print(f"  MSE  (均方误差):      {results['MSE']:.4f}")
    print(f"  MAE  (平均绝对误差):  {results['MAE']:.4f}")
    print(f"  PSNR (峰值信噪比):    {results['PSNR']:.2f} dB")
    print(f"  SNR  (信噪比):        {results['SNR']:.2f} dB")
    print(f"  SSIM (结构相似性):    {results['SSIM']:.4f}")
    print(f"  NCC  (归一化互相关):  {results['NCC']:.4f}")
    print("-" * 50)
    print(f"  综合评级: {grade}")
    print("=" * 50)

    return results


# ==========================================
# 批量评价
# ==========================================

def batch_evaluate(original_folder, stego_folder, prefix="stego_"):
    """
    批量评价文件夹中的图片
    original_folder: 原图文件夹
    stego_folder: stego 图文件夹
    prefix: stego 文件名前缀
    """
    # 获取所有原图
    original_files = glob(os.path.join(original_folder, "*.png"))
    original_files += glob(os.path.join(original_folder, "*.jpg"))
    original_files = sorted(original_files)

    if not original_files:
        print(f"错误: 在 {original_folder} 找不到图片")
        return None

    all_results = []

    print("\n" + "=" * 70)
    print("  批量图像质量评价")
    print("=" * 70)
    print(f"原图文件夹: {original_folder}")
    print(f"Stego文件夹: {stego_folder}")
    print(f"找到 {len(original_files)} 张原图")
    print("-" * 70)
    print(f"{'文件名':<30} | {'PSNR':>8} | {'SSIM':>8} | {'MSE':>10}")
    print("-" * 70)

    for orig_path in original_files:
        filename = os.path.basename(orig_path)
        stego_filename = prefix + filename
        stego_path = os.path.join(stego_folder, stego_filename)

        if not os.path.exists(stego_path):
            print(f"{filename:<30} | {'跳过 - 无对应stego'}")
            continue

        # 读取图片
        original = cv2.imread(orig_path)
        stego = cv2.imread(stego_path)

        if original is None or stego is None:
            print(f"{filename:<30} | {'跳过 - 读取失败'}")
            continue

        # 计算指标
        results = evaluate_image_quality(original, stego)
        results['filename'] = filename
        all_results.append(results)

        print(f"{filename:<30} | {results['PSNR']:>7.2f}dB | {results['SSIM']:>8.4f} | {results['MSE']:>10.2f}")

    # 统计汇总
    if all_results:
        avg_psnr = np.mean([r['PSNR'] for r in all_results])
        avg_ssim = np.mean([r['SSIM'] for r in all_results])
        avg_mse = np.mean([r['MSE'] for r in all_results])

        print("-" * 70)
        print(f"{'平均值':<30} | {avg_psnr:>7.2f}dB | {avg_ssim:>8.4f} | {avg_mse:>10.2f}")
        print("=" * 70)

        grade = get_quality_grade(avg_psnr, avg_ssim)
        print(f"综合评级: {grade}")

        return {
            'details': all_results,
            'average': {
                'PSNR': avg_psnr,
                'SSIM': avg_ssim,
                'MSE': avg_mse,
            },
            'grade': grade
        }

    return None


def compare_single_image(original_path, stego_path):
    """比较单张图片"""
    original = cv2.imread(original_path)
    stego = cv2.imread(stego_path)

    if original is None:
        print(f"错误: 无法读取原图 {original_path}")
        return None

    if stego is None:
        print(f"错误: 无法读取stego图 {stego_path}")
        return None

    if original.shape != stego.shape:
        print(f"警告: 图片尺寸不匹配")
        print(f"  原图: {original.shape}")
        print(f"  Stego: {stego.shape}")
        # 尝试调整大小
        stego = cv2.resize(stego, (original.shape[1], original.shape[0]))

    return print_quality_report(original, stego, os.path.basename(stego_path))


# ==========================================
# 主程序
# ==========================================

def main():
    print("\n" + "=" * 50)
    print("  图像质量评价系统")
    print("=" * 50)
    print("\n请选择模式:")
    print("  [1] 单张图片评价")
    print("  [2] 批量评价（文件夹）")
    print("  [3] 快速测试（内置路径）")

    choice = input("\n请选择 (1/2/3): ").strip()

    if choice == '1':
        original_path = input("请输入原图路径: ").strip()
        stego_path = input("请输入stego图路径: ").strip()
        compare_single_image(original_path, stego_path)

    elif choice == '2':
        original_folder = input("请输入原图文件夹路径: ").strip()
        stego_folder = input("请输入stego文件夹路径: ").strip()
        prefix = input("stego文件名前缀 (默认 'stego_'): ").strip()
        if not prefix:
            prefix = "stego_"
        batch_evaluate(original_folder, stego_folder, prefix)

    elif choice == '3':
        # 快速测试：使用默认路径
        original_folder = r"D:\paper data\output_3\I"
        stego_folder = r"D:\paper data\stego_output"
        batch_evaluate(original_folder, stego_folder, "stego_")

    else:
        print("无效选择")


if __name__ == "__main__":
    main()