# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import qimtest
from glob import glob

# =================配置区域=================
IMG_FOLDER = r"D:\paper data\output_3\10"  # 图片文件夹
PATCH_SIZE = 128
CONTEXT_MULT = 6.0
QIM_STEP = 200
NUM_POINTS = 20  # 每张图测试的点数
PAYLOAD_SIZE = 31


def get_patch_transform(kp, output_size):
    x, y = kp.pt
    angle = kp.angle
    size = kp.size
    scale_factor = output_size / (size * CONTEXT_MULT)
    M = cv2.getRotationMatrix2D((x, y), -angle, scale_factor)
    M[0, 2] += (output_size / 2) - x
    M[1, 2] += (output_size / 2) - y
    return M


def embed_packet_into_patch(patch, packet_bytes):
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


def test_single_image(img_path):
    """测试单张图片，返回 (总点数, 存活点数)"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0, 0

    h, w = img.shape

    # 检测并过滤特征点
    sift = cv2.SIFT_create()
    kp_raw = sift.detect(img, None)

    kp_filtered = []
    for kp in kp_raw:
        angle = kp.angle
        scale_factor = PATCH_SIZE / (kp.size * CONTEXT_MULT)
        angle_ok = angle < 5 or angle > 355
        scale_ok = 0.9 < scale_factor < 1.1
        if angle_ok and scale_ok:
            kp_filtered.append(kp)

    kp_selected = sorted(kp_filtered, key=lambda x: -x.response)[:NUM_POINTS]

    if len(kp_selected) == 0:
        return 0, 0

    # 模拟嵌入并测试存活
    dummy_data = b'\xAA' * PAYLOAD_SIZE
    survived = 0

    for kp in kp_selected:
        try:
            M = get_patch_transform(kp, PATCH_SIZE)
            patch = cv2.warpAffine(img, M, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_NEAREST)
            patch_stego = embed_packet_into_patch(patch, dummy_data)

            # 放回
            temp_img = img.copy()
            M_inv = cv2.invertAffineTransform(M)
            patch_back = cv2.warpAffine(patch_stego, M_inv, (w, h), flags=cv2.INTER_NEAREST)
            mask = np.full((PATCH_SIZE, PATCH_SIZE), 255, dtype=np.uint8)
            mask_warped = cv2.warpAffine(mask, M_inv, (w, h), flags=cv2.INTER_NEAREST)
            region = (mask_warped > 10)
            temp_img[region] = patch_back[region]

            # 提取验证
            patch_verify = cv2.warpAffine(temp_img, M, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_NEAREST)
            extracted = extract_packet_from_patch(patch_verify)

            if extracted == dummy_data:
                survived += 1
        except:
            continue

    return len(kp_selected), survived


def main():
    # 获取所有图片
    img_files = glob(os.path.join(IMG_FOLDER, "*.png"))
    img_files += glob(os.path.join(IMG_FOLDER, "*.jpg"))

    print(f"=== 批量存活率测试 ===")
    print(f"文件夹: {IMG_FOLDER}")
    print(f"图片数: {len(img_files)}")
    print(f"配置: PATCH={PATCH_SIZE}, STEP={QIM_STEP}, 每图测{NUM_POINTS}点")
    print("-" * 60)

    total_points = 0
    total_survived = 0
    results = []

    for i, img_path in enumerate(img_files):
        pts, surv = test_single_image(img_path)
        total_points += pts
        total_survived += surv
        rate = surv / pts * 100 if pts > 0 else 0
        results.append((os.path.basename(img_path), pts, surv, rate))

        # 每10张打印一次进度
        if (i + 1) % 10 == 0:
            print(f"进度: {i + 1}/{len(img_files)}")

    # 汇总
    print("-" * 60)
    print(f"{'图片':<30} | {'点数':<6} | {'存活':<6} | {'存活率'}")
    print("-" * 60)

    for name, pts, surv, rate in results:
        print(f"{name:<30} | {pts:<6} | {surv:<6} | {rate:.1f}%")

    print("=" * 60)
    avg_rate = total_survived / total_points * 100 if total_points > 0 else 0
    print(f"总计: {total_points} 点, 存活 {total_survived} 点")
    print(f"平均存活率: {avg_rate:.1f}%")


if __name__ == "__main__":
    main()