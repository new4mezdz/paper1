# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# =================配置区域=================
# =================配置区域=================
IMG_PATH = r"D:\paper data\stego_images\I_pts_364.png"
TEMPLATE_RADIUS = 90
TEMPLATE_ANGLE = 30
TEMPLATE_STRENGTH = 500 # 改成5000万（增加10倍）

# ==========================================

def get_spectrum_vis(img):
    """获取可视化的频谱图"""
    f = np.fft.fft2(img.astype(float))
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def add_dft_template(img, radius=90, strength=5000000):
    """新模板：在30度方向上放3个点（非对称，易检测）"""
    h, w = img.shape
    cx, cy = w // 2, h // 2
    f = np.fft.fft2(img.astype(float))
    fshift = np.fft.fftshift(f)

    # 在30度方向上放3个不同半径的点
    rad = np.deg2rad(TEMPLATE_ANGLE)

    radii = [70, 90, 110]  # 三个不同半径
    for r in radii:
        off_x = int(r * np.cos(rad))
        off_y = int(r * np.sin(rad))

        # 每个点都是对称的（中心对称）
        p1 = (cx + off_x, cy + off_y)
        p2 = (cx - off_x, cy - off_y)

        fshift[p1[1], p1[0]] += strength
        fshift[p2[1], p2[0]] += strength

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    return np.clip(img_back, 0, 255).astype(np.uint8)

# ==========================================
# 🚑 核心功能：自动几何校正
# ==========================================
def geometric_correction(img_attacked, original_radius, original_angle):
    """改进：检测三点一线的模板"""
    h, w = img_attacked.shape
    cx, cy = w // 2, h // 2

    f = np.fft.fft2(img_attacked.astype(float))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # 抹除中心
    mask_radius = 40
    y_grid, x_grid = np.ogrid[:h, :w]
    center_mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= mask_radius ** 2
    magnitude[center_mask] = 0

    print("\n[侦探] 寻找三点模板...")

    # 创建角度+半径掩码（搜索30度方向，半径60-120）
    dy_grid = y_grid - cy
    dx_grid = x_grid - cx
    angle_grid = np.degrees(np.arctan2(dy_grid, dx_grid))
    radius_grid = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

    # 30度方向 ±20度，半径60-120
    angle_mask = np.abs(angle_grid - 30) <= 20
    radius_mask = (radius_grid >= 60) & (radius_grid <= 120)

    # 对称方向（-150度）
    angle_mask2 = np.abs(angle_grid - (-150)) <= 20

    search_mask = (angle_mask | angle_mask2) & radius_mask

    # 找峰值
    masked_mag = magnitude.copy()
    masked_mag[~search_mask] = 0

    if masked_mag.max() > 0:
        max_idx = np.unravel_index(np.argmax(masked_mag), masked_mag.shape)
        found_y, found_x = max_idx

        dy = found_y - cy
        dx = found_x - cx
        current_radius = np.sqrt(dx ** 2 + dy ** 2)
        current_angle = np.degrees(np.arctan2(dy, dx))

        print(f"  检测到: 半径={current_radius:.1f}, 角度={current_angle:.1f}°")

        # 计算旋转量
        diff = current_angle - original_angle

        # 归一化到[-45, 45]
        while diff < -45:
            diff += 90
        while diff > 45:
            diff -= 90

        rotation = -diff
        scale = 90 / current_radius  # 用中间点（90）作基准

        print(f"[修复] 旋转={rotation:.1f}°, 缩放={scale:.2f}x")

        # 生成候选（现在三点模板有方向性，但仍可能180度歧义）
        candidates = []
        for extra in [0, 180]:  # 只需要2个候选
            angle = rotation + extra
            M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
            recovered = cv2.warpAffine(img_attacked, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderValue=128)
            candidates.append(recovered)

        # 补齐到4个（为了兼容显示代码）
        candidates.extend([candidates[0], candidates[1]])

        return candidates
    else:
        print("[错误] 未检测到模板！")
        return [img_attacked] * 4

# ==========================================
# ⚔️ 攻击函数 (保持不变)
# ==========================================
def attack_combo(img, angle, scale):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
    return cv2.warpAffine(img, M, (w, h), borderValue=128)


def interactive_attack():
    """交互式选择攻击类型和参数"""
    print("\n" + "=" * 50)
    print("  ⚔️  几何攻击菜单")
    print("=" * 50)
    print("1. 旋转 (Rotation)")
    print("2. 缩放 (Scaling)")
    print("3. 裁剪 (Cropping)")
    print("4. 旋转+缩放组合")
    print("5. 全部攻击组合")

    choice = input("\n请选择攻击类型 (1-5): ").strip()

    attacks = {}

    if choice in ['1', '4', '5']:
        angle = float(input("  输入旋转角度 (度, 如30): "))
        attacks['rotate'] = angle

    if choice in ['2', '4', '5']:
        scale = float(input("  输入缩放比例 (如0.8表示缩小到80%): "))
        attacks['scale'] = scale

    if choice in ['3', '5']:
        crop = float(input("  输入裁剪比例 (如0.1表示裁掉10%): "))
        attacks['crop'] = crop

    return attacks


def apply_attacks(img, attacks):
    """应用选定的攻击"""
    h, w = img.shape
    result = img.copy()

    # 裁剪
    if 'crop' in attacks:
        ratio = attacks['crop']
        crop_size = int(min(h, w) * (1 - ratio))
        start = (h - crop_size) // 2
        result = result[start:start + crop_size, start:start + crop_size]
        result = cv2.resize(result, (w, h))

    # 旋转+缩放
    angle = attacks.get('rotate', 0)
    scale = attacks.get('scale', 1.0)

    if angle != 0 or scale != 1.0:
        # === 改进：计算旋转后需要的画布大小 ===
        if angle != 0 and scale == 1.0:  # 纯旋转，保留所有内容
            # 计算旋转后的边界框
            rad = np.deg2rad(abs(angle))
            new_w = int(h * np.sin(rad) + w * np.cos(rad))
            new_h = int(h * np.cos(rad) + w * np.sin(rad))

            # 调整变换矩阵，将旋转中心移到新画布中心
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            result = cv2.warpAffine(result, M, (new_w, new_h), borderValue=128)
            # 缩回原尺寸以便比较
            result = cv2.resize(result, (w, h))
        else:
            # 正常的旋转+缩放（会裁剪边缘）
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
            result = cv2.warpAffine(result, M, (w, h), borderValue=128)

    return result

def main():
    if not os.path.exists(IMG_PATH):
        print(f"错误: 找不到 {IMG_PATH}")
        return

    # 1. 准备
    img_orig = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    img_orig = cv2.resize(img_orig, (512, 512))
    img_template = add_dft_template(img_orig, TEMPLATE_RADIUS, TEMPLATE_STRENGTH)

    # 2. 交互式选择攻击
    attacks = interactive_attack()
    img_attacked = apply_attacks(img_template, attacks)

    print(f"\n已应用攻击: {attacks}")

    # 3. 自动恢复（获取4个候选）
    candidates = geometric_correction(img_attacked, TEMPLATE_RADIUS, TEMPLATE_ANGLE)

    # 4. 可视化（2行6列布局）
    plt.figure(figsize=(24, 8))

    # 第一行：原图、攻击图、4个候选
    plt.subplot(2, 6, 1)
    plt.imshow(img_template, cmap='gray')
    plt.title("1. Original", fontsize=12)
    plt.axis('off')

    plt.subplot(2, 6, 2)
    plt.imshow(img_attacked, cmap='gray')
    plt.title(f"2. Attacked\n{attacks}", fontsize=10, color='red')
    plt.axis('off')

    for i in range(4):
        plt.subplot(2, 6, 3 + i)
        plt.imshow(candidates[i], cmap='gray')
        plt.title(f"候选{i + 1}\n(+{i * 90}°)", fontsize=11, color='green')
        plt.axis('off')

    # 第二行：对应的频谱
    plt.subplot(2, 6, 7)
    plt.imshow(get_spectrum_vis(img_template), cmap='gray')
    plt.title("Original Spectrum", fontsize=9)
    plt.axis('off')

    plt.subplot(2, 6, 8)
    plt.imshow(get_spectrum_vis(img_attacked), cmap='gray')
    plt.title("Attacked Spectrum", fontsize=9)
    plt.axis('off')

    for i in range(4):
        plt.subplot(2, 6, 9 + i)
        plt.imshow(get_spectrum_vis(candidates[i]), cmap='gray')
        plt.title(f"Spectrum {i + 1}", fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()