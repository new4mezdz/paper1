# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =================配置区域=================
IMG_PATH = r"D:\paper data\output_3\I\I_pts_0.png"
ATTACK_ANGLE = 45  # 旋转角度
CROP_PERCENT = 20  # 裁剪掉四周 20%
TOP_K = 4  # 只展示最强的 4 个匹配点


# ==========================================

def apply_attack_with_info(img, angle, crop_pct):
    """
    实施攻击，并返回'变换矩阵'和'裁剪偏移量'，用于后续数学验证
    """
    h, w = img.shape[:2]

    # 1. 旋转
    # 获取旋转矩阵 (2x3)
    M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M_rot, (w, h), borderValue=(128, 128, 128))

    # 2. 裁剪
    margin_h = int(h * (crop_pct / 100))
    margin_w = int(w * (crop_pct / 100))
    cropped = rotated[margin_h:h - margin_h, margin_w:w - margin_w]

    # 返回: 攻击后的图, 旋转矩阵, 裁剪掉的宽(x偏移), 裁剪掉的高(y偏移)
    return cropped, M_rot, margin_w, margin_h


def verify_point_location(pt_orig, pt_attacked, M, off_x, off_y):
    """
    数学验证：计算原图的点经过旋转裁剪后，理论上应该在哪里
    """
    # 1. 旋转坐标映射
    # P_new = M * P_old (齐次坐标)
    px, py = pt_orig
    # 矩阵乘法: x' = M00*x + M01*y + M02
    #           y' = M10*x + M11*y + M12
    rot_x = M[0, 0] * px + M[0, 1] * py + M[0, 2]
    rot_y = M[1, 0] * px + M[1, 1] * py + M[1, 2]

    # 2. 减去裁剪偏移
    theory_x = rot_x - off_x
    theory_y = rot_y - off_y

    # 3. 计算与实际识别点的距离误差
    actual_x, actual_y = pt_attacked
    error = np.sqrt((theory_x - actual_x) ** 2 + (theory_y - actual_y) ** 2)

    return error, (theory_x, theory_y)


def main():
    if not os.path.exists(IMG_PATH):
        print(f"错误: 找不到 {IMG_PATH}")
        return

    # 1. 读取与攻击
    img_orig = cv2.imread(IMG_PATH)
    # 获取攻击后的图以及变换参数(用于验证)
    img_attacked, M, off_x, off_y = apply_attack_with_info(img_orig, ATTACK_ANGLE, CROP_PERCENT)

    print(f"=== SIFT 特征点一致性验证 ===")
    print(f"攻击参数: 旋转 {ATTACK_ANGLE}° | 裁剪 {CROP_PERCENT}%")

    # 2. SIFT 提取 (检测 + 计算描述符)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_orig, None)
    kp2, des2 = sift.detectAndCompute(img_attacked, None)

    if des1 is None or des2 is None:
        print("错误: 未检测到足够的特征点")
        return

    # 3. 特征匹配 (使用暴力匹配器 BFMatcher)
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)

    # 按距离排序，只取最相似的前 TOP_K 个
    matches = sorted(matches, key=lambda x: x.distance)[:TOP_K]

    print(f"\n[验证报告] 检查前 {len(matches)} 个最佳匹配点:")
    print("-" * 60)
    print(f"{'ID':<4} | {'误差(像素)':<12} | {'判定结果':<10} | {'说明'}")
    print("-" * 60)

    # 4. 逐个验证并打印结果
    valid_matches = []

    for i, m in enumerate(matches):
        # 获取原图中的点坐标
        pt_orig = kp1[m.queryIdx].pt
        # 获取攻击后图中的点坐标
        pt_attacked = kp2[m.trainIdx].pt

        # 数学计算：理论上它应该在哪里？
        error, pt_theory = verify_point_location(pt_orig, pt_attacked, M, off_x, off_y)

        # 判定 (误差小于 3 像素认为极其精准)
        if error < 3.0:
            status = "✅ 成功"
            desc = "完美重合"
        elif error < 10.0:
            status = "⚠️ 偏差"
            desc = "稍微偏移"
        else:
            status = "❌ 失败"
            desc = "位置错误"

        print(
            f"#{i + 1:<3} | {error:.2f} px       | {status}    | 理论:({pt_theory[0]:.1f},{pt_theory[1]:.1f}) -> 实际:({pt_attacked[0]:.1f},{pt_attacked[1]:.1f})")
        valid_matches.append(m)

    # 5. 可视化连线
    # drawMatches 会自动把两张图拼在一起，并画线连接对应的点
    img_matches = cv2.drawMatches(img_orig, kp1, img_attacked, kp2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                  matchColor=(0, 255, 0),  # 绿色连线
                                  singlePointColor=None)

    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f"SIFT Matching Verification (Top {TOP_K})\nLines indicate matched features despite Rotation & Crop",
              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()