# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import qimtest

# =================配置区域=================
IMG_PATH = r"D:\paper data\output_3\I\I_pts_90090.png"
PATCH_SIZE = 128
CONTEXT_MULT = 6.0
QIM_STEP = 200
NUM_POINTS = 20
PAYLOAD_SIZE = 31


# ==========================================

def get_patch_transform(kp, output_size):
    """计算从原图到标准化补丁的变换矩阵"""
    x, y = kp.pt
    angle = kp.angle
    size = kp.size
    scale_factor = output_size / (size * CONTEXT_MULT)

    M = cv2.getRotationMatrix2D((x, y), -angle, scale_factor)
    M[0, 2] += (output_size / 2) - x
    M[1, 2] += (output_size / 2) - y
    return M


def embed_packet_into_patch(patch, packet_bytes):
    """嵌入数据到patch"""
    bits = qimtest.bytes_to_bits(packet_bytes)
    if len(bits) > 252:
        bits = bits[:252]

    stego_patch = patch.astype(float)
    bit_idx = 0
    h, w = patch.shape
    center_blocks = [(7, 7), (7, 8), (8, 7), (8, 8)]

    for y in range(0, h, 8):
        for x in range(0, w, 8):
            if bit_idx >= len(bits):
                break
            if (y // 8, x // 8) in center_blocks:
                continue
            block = stego_patch[y:y + 8, x:x + 8]
            dct_block = cv2.dct(block)
            dct_block[4, 3] = qimtest.qim_embed_scalar(dct_block[4, 3], bits[bit_idx], QIM_STEP)
            stego_patch[y:y + 8, x:x + 8] = cv2.idct(dct_block)
            bit_idx += 1

    return np.clip(stego_patch, 0, 255).astype(np.uint8)


def extract_packet_from_patch(patch):
    """从patch提取数据"""
    patch = patch.astype(float)
    bits = []
    h, w = patch.shape
    center_blocks = [(7, 7), (7, 8), (8, 7), (8, 8)]

    for y in range(0, h, 8):
        for x in range(0, w, 8):
            if (y // 8, x // 8) in center_blocks:
                continue
            block = patch[y:y + 8, x:x + 8]
            if block.shape != (8, 8):
                continue
            dct_block = cv2.dct(block)
            bit = qimtest.qim_extract_scalar(dct_block[4, 3], QIM_STEP)
            bits.append(bit)

    needed_bits = PAYLOAD_SIZE * 8
    bits = bits[:needed_bits]
    while len(bits) < needed_bits:
        bits.append(0)

    return qimtest.bits_to_bytes(np.array(bits, dtype=np.uint8))


def embed_simulation(img_gray, keypoints):
    """模拟嵌入过程"""
    stego_img = img_gray.copy()
    h, w = img_gray.shape

    dummy_data = b'\xAA' * PAYLOAD_SIZE
    print(f"[测试] 包大小: {len(dummy_data)} 字节")

    for i, kp in enumerate(keypoints):
        try:
            M = get_patch_transform(kp, PATCH_SIZE)
            # 改用 INTER_NEAREST
            patch = cv2.warpAffine(stego_img, M, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_NEAREST)

            patch_stego = embed_packet_into_patch(patch, dummy_data)

            # 调试：嵌入后立即提取验证
            extracted_immediately = extract_packet_from_patch(patch_stego)
            immediate_status = "✅" if extracted_immediately == dummy_data else "❌"

            M_inv = cv2.invertAffineTransform(M)
            # 改用 INTER_NEAREST
            patch_back = cv2.warpAffine(patch_stego, M_inv, (w, h), flags=cv2.INTER_NEAREST)

            white_patch = np.full((PATCH_SIZE, PATCH_SIZE), 255, dtype=np.uint8)
            # 改用 INTER_NEAREST
            mask = cv2.warpAffine(white_patch, M_inv, (w, h), flags=cv2.INTER_NEAREST)

            region = (mask > 10)
            stego_img[region] = patch_back[region]

            # 调试：放回后再提取验证
            patch_after = cv2.warpAffine(stego_img, M, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_NEAREST)
            extracted_after = extract_packet_from_patch(patch_after)
            after_status = "✅" if extracted_after == dummy_data else "❌"

            print(f"  点#{i + 1}: 嵌入后立即提取{immediate_status}, 放回后提取{after_status}")

        except Exception as e:
            print(f"嵌入点 {kp.pt} 失败: {e}")

    return stego_img


def main():
    if not os.path.exists(IMG_PATH):
        print(f"错误: 找不到 {IMG_PATH}")
        return

    # 1. 准备阶段
    img_bgr = cv2.imread(IMG_PATH)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. 初始检测
    sift = cv2.SIFT_create()
    kp_raw = sift.detect(img_gray, None)
    kp_filtered = []
    for kp in kp_raw:
        angle = kp.angle
        scale_factor = PATCH_SIZE / (kp.size * CONTEXT_MULT)

        # 更严格：角度 < 3度，缩放误差 < 5%
        angle_ok = angle < 3 or angle > 357
        scale_ok = 0.95 < scale_factor < 1.05

        if angle_ok and scale_ok:
            kp_filtered.append(kp)

    kp_orig = sorted(kp_filtered, key=lambda x: -x.response)[:NUM_POINTS]
    # 3. 执行嵌入
    print("正在嵌入数据...")
    stego_img = embed_simulation(img_gray, kp_orig)

    # 4. 再次检测
    print("正在重新检测特征点...")
    kp_new = sift.detect(stego_img, None)

    # 5. 三重验证：位置 + 角度/尺度 + 数据
    survived_kps = []
    threshold_dist = 3.0
    threshold_angle = 5.0
    threshold_size = 2.0

    original_data = b'\xAA' * PAYLOAD_SIZE

    print("\n[存活详情]")
    print(f"{'ID':<4} | {'原坐标':<20} | {'位置':<4} | {'参数':<4} | {'数据':<4} | {'详情'}")
    print("-" * 80)

    pos_ok_count = 0
    param_ok_count = 0
    data_ok_count = 0

    for i, old_k in enumerate(kp_orig):
        best_match = None
        min_dist = 99999

        for new_k in kp_new:
            dist = np.sqrt((old_k.pt[0] - new_k.pt[0]) ** 2 + (old_k.pt[1] - new_k.pt[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_match = new_k

        # 位置检查
        pos_ok = min_dist < threshold_dist

        # 角度/尺度检查
        param_ok = False
        angle_diff = 0
        size_diff = 0
        if best_match and pos_ok:
            angle_diff = abs(old_k.angle - best_match.angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            size_diff = abs(old_k.size - best_match.size)
            param_ok = (angle_diff < threshold_angle and size_diff < threshold_size)

        # 数据提取检查（改用 INTER_NEAREST）
        data_ok = False
        if pos_ok:
            M = get_patch_transform(old_k, PATCH_SIZE)
            patch = cv2.warpAffine(stego_img, M, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_NEAREST)
            extracted = extract_packet_from_patch(patch)
            data_ok = (extracted == original_data)

        # 统计
        if pos_ok:
            pos_ok_count += 1
        if param_ok:
            param_ok_count += 1
        if data_ok:
            data_ok_count += 1

        # 输出
        pos_str = "✅" if pos_ok else "❌"
        param_str = "✅" if param_ok else "❌"
        data_str = "✅" if data_ok else "❌"
        detail = f"dist={min_dist:.1f}, Δang={angle_diff:.1f}°, Δsize={size_diff:.1f}" if pos_ok else f"dist={min_dist:.1f}"

        print(f"#{i + 1:<3} | ({old_k.pt[0]:6.1f}, {old_k.pt[1]:6.1f}) | {pos_str:<4} | {param_str:<4} | {data_str:<4} | {detail}")

        if pos_ok and param_ok and data_ok:
            survived_kps.append(old_k)

    # 6. 汇总
    print("-" * 80)
    print(f"位置存活: {pos_ok_count}/{len(kp_orig)}")
    print(f"参数存活: {param_ok_count}/{len(kp_orig)}")
    print(f"数据正确: {data_ok_count}/{len(kp_orig)}")
    print(f"完全存活: {len(survived_kps)}/{len(kp_orig)} ({len(survived_kps) / len(kp_orig) * 100:.1f}%)")

    # 7. 可视化
    plt.figure(figsize=(12, 8))

    vis_orig = cv2.drawKeypoints(img_bgr, kp_orig, None, color=(0, 0, 255),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for kp in survived_kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(vis_orig, (x, y), int(kp.size / 2), (0, 255, 0), 2)
        cv2.putText(vis_orig, "OK", (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB))
    plt.title(f"Survival Test (Step={QIM_STEP}, INTER_NEAREST)\nRed=All, Green=Survived ({len(survived_kps)}/{len(kp_orig)})", fontsize=14)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()