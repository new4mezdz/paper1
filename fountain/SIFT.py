# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =================配置区域=================
IMG_PATH = r"D:\paper data\output_3\I\I_pts_0.png"
PATCH_SIZE = 64  # 补丁大小
NUM_POINTS = 4  # 展示前 4 个特征点
CONTEXT_MULT = 20.0  # 【关键】扩大视野倍数 (越大看到的纹理越多)


# ==========================================

def get_patch_transform(kp, output_size, context_mult):
    """
    计算从'原图局部'到'标准化补丁'的仿射变换矩阵 M
    """
    x, y = kp.pt
    angle = kp.angle
    size = kp.size

    # 计算缩放比例：我们要包含 size * context_mult 这么大的区域
    scale_factor = output_size / (size * context_mult)

    # 1. 获取旋转矩阵 (以特征点为中心)
    M = cv2.getRotationMatrix2D((x, y), -angle, scale_factor)

    # 2. 修正平移 (把特征点移到补丁中心)
    M[0, 2] += (output_size / 2) - x
    M[1, 2] += (output_size / 2) - y

    return M


def main():
    if not os.path.exists(IMG_PATH):
        print(f"错误: 找不到 {IMG_PATH}")
        return

    # 1. 读取图片
    img_bgr = cv2.imread(IMG_PATH)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. SIFT 检测
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img_gray, None)
    keypoints = sorted(keypoints, key=lambda x: -x.response)[:NUM_POINTS]

    print(f"提取前 {len(keypoints)} 个特征点，视野放大 {CONTEXT_MULT} 倍...")

    # 3. 准备可视化大图
    img_vis = img_bgr.copy()

    # 用于存放提取出的补丁
    patches = []

    # 4. 核心循环：处理每个点
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # 红, 绿, 蓝, 黄

    for i, kp in enumerate(keypoints):
        color = colors[i % len(colors)]

        # A. 计算变换矩阵 M
        M = get_patch_transform(kp, PATCH_SIZE, CONTEXT_MULT)

        # B. 提取补丁 (正向变换)
        patch = cv2.warpAffine(img_gray, M, (PATCH_SIZE, PATCH_SIZE),
                               flags=cv2.INTER_LINEAR)
        patches.append(patch)

        # C. 【关键】在原图上画出这个补丁的范围 (逆向变换)
        # 定义补丁的四个角 (0,0) -> (64,0) -> (64,64) -> (0,64)
        h, w = PATCH_SIZE, PATCH_SIZE
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

        # 计算逆矩阵
        M_inv = cv2.invertAffineTransform(M)

        # 将四个角映射回原图坐标
        src_pts = cv2.transform(pts, M_inv)

        # 画出旋转矩形框
        cv2.polylines(img_vis, [np.int32(src_pts)], True, color, 3, cv2.LINE_AA)

        # 标上序号
        text_pos = np.int32(src_pts[0][0])
        cv2.putText(img_vis, f"#{i + 1}", (text_pos[0], text_pos[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 5. 绘图
    plt.figure(figsize=(12, 8))

    # 上半部分：原图 + 框
    plt.subplot(2, 4, (1, 4))
    # BGR 转 RGB 用于 matplotlib 显示
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Global View (Boxes = Normalized Regions)", fontsize=14)
    plt.axis('off')

    # 下半部分：对应的补丁
    for i, patch in enumerate(patches):
        plt.subplot(2, 4, 5 + i)
        plt.imshow(patch, cmap='gray')

        # 给补丁加个对应颜色的边框，方便看
        # matplotlib 的边框设置稍微麻烦点，这里用标题颜色区分
        c_name = ['Red', 'Green', 'Blue', 'Yellow'][i % 4]
        plt.title(f"Patch #{i + 1} ({c_name})\n(Standard 64x64)", fontsize=12, color=c_name.lower())
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()