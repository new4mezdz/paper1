# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import lt_min
import qimtest

# =================配置区域=================
IMG_PATH = r"D:\paper data\output_3\I\I_pts_0.png"
PAYLOAD_SIZE = 31  # 序列化后的总包大小 (8字节头部 + 23字节payload)
BLOCK_SIZE_FOR_LT = 23  # LT编码的实际payload大小
PATCH_SIZE = 128  # 补丁大小
QIM_STEP = 200  # 步长

NUM_IMAGES = 1  # 图片数量
PACKETS_PER_IMG = 10  # 最终每张图要嵌多少个包
CANDIDATE_POOL = 100  # 海选池
SEARCH_RANGE = 500  # 提取时扫描前 500 个点


# ==========================================

def get_patch_transform(kp, output_size):
    """只平移，不旋转不缩放"""
    x, y = kp.pt
    M = np.float32([
        [1, 0, output_size / 2 - x],
        [0, 1, output_size / 2 - y]
    ])
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

    bits = np.array(bits, dtype=np.uint8)
    return qimtest.bits_to_bytes(bits)


def filter_keypoints_by_boundary(kps_all, img_shape):
    """过滤：只保留离边界足够远的点"""
    h, w = img_shape
    half = PATCH_SIZE // 2
    margin = half + 10  # 留点余量

    kp_filtered = []
    for kp in kps_all:
        x, y = kp.pt
        if margin < x < w - margin and margin < y < h - margin:
            kp_filtered.append(kp)

    return kp_filtered


def filter_stable_keypoints(img_gray, candidates):
    """数据完整性过滤：只保留能正确提取数据的点"""
    h, w = img_gray.shape
    dummy_data = b'\xAA' * PAYLOAD_SIZE

    survivors = []

    for kp in candidates:
        try:
            # 1. 提取 patch
            M = get_patch_transform(kp, PATCH_SIZE)
            patch = cv2.warpAffine(img_gray, M, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_NEAREST)

            # 2. 嵌入数据
            patch_stego = embed_packet_into_patch(patch, dummy_data)

            # 3. 放回原图
            temp_img = img_gray.copy()
            M_inv = cv2.invertAffineTransform(M)
            patch_back = cv2.warpAffine(patch_stego, M_inv, (w, h), flags=cv2.INTER_NEAREST)
            mask = np.full((PATCH_SIZE, PATCH_SIZE), 255, dtype=np.uint8)
            mask_warped = cv2.warpAffine(mask, M_inv, (w, h), flags=cv2.INTER_NEAREST)
            region = (mask_warped > 10)
            temp_img[region] = patch_back[region]

            # 4. 重新提取验证
            patch_verify = cv2.warpAffine(temp_img, M, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_NEAREST)
            extracted = extract_packet_from_patch(patch_verify)

            # 5. 只有数据完全正确才保留
            if extracted == dummy_data:
                survivors.append(kp)

        except:
            continue

    return survivors


def main():
    if not os.path.exists(IMG_PATH):
        print("找不到图片")
        return

    secret_msg = b"Hajimi-sama's Robust System" * 5

    # 使用 BLOCK_SIZE_FOR_LT (23字节)
    encoder = lt_min.LTEncoder(secret_msg, block_size=BLOCK_SIZE_FOR_LT, base_seed=2025)

    # 验证包大小
    test_pkt = encoder.next_packet()
    test_bytes = lt_min.serialize_lt_packet(test_pkt)
    print(f"\n[验证] LT包大小: payload={BLOCK_SIZE_FOR_LT}字节, 序列化后={len(test_bytes)}字节")
    if len(test_bytes) != PAYLOAD_SIZE:
        print(f"[错误] 包大小不匹配! 期望{PAYLOAD_SIZE}, 实际{len(test_bytes)}")
        return

    print(f"[信息] 消息长度={len(secret_msg)}字节, 需要k={encoder.k}个包")

    img_bgr = cv2.imread(IMG_PATH)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    stego_images = []

    print(f"\n=== 简化版：只平移，不旋转不缩放 ===")
    print(f"配置: QIM_STEP={QIM_STEP}, PATCH_SIZE={PATCH_SIZE}")

    # [Phase 1] 生成
    for i in range(NUM_IMAGES):
        packets = [lt_min.serialize_lt_packet(encoder.next_packet()) for _ in range(PACKETS_PER_IMG)]

        # 检测特征点
        sift = cv2.SIFT_create()
        kps_all = sift.detect(img_gray, None)
        print(f"\n图片 #{i + 1}: 检测到 {len(kps_all)} 个特征点")

        # 第一轮：边界过滤
        kps_boundary = filter_keypoints_by_boundary(kps_all, img_gray.shape)
        print(f"  边界过滤后: {len(kps_boundary)} 个")

        # 按响应排序，取前 CANDIDATE_POOL 个
        candidates = sorted(kps_boundary, key=lambda x: -x.response)[:CANDIDATE_POOL]
        print(f"  候选池: {len(candidates)} 个")

        # 第二轮：数据完整性过滤
        stable_kps = filter_stable_keypoints(img_gray, candidates)
        print(f"  数据完整性过滤后: {len(stable_kps)} 个")

        target_kps = stable_kps[:PACKETS_PER_IMG]
        print(f"  最终使用: {len(target_kps)} 个")

        if len(target_kps) < PACKETS_PER_IMG:
            print(f"  ⚠️ 警告: 只有 {len(target_kps)} 个稳定点，少于需求 {PACKETS_PER_IMG}")

        # 嵌入
        current_stego = img_gray.copy()
        for idx, kp in enumerate(target_kps):
            M = get_patch_transform(kp, PATCH_SIZE)
            patch = cv2.warpAffine(current_stego, M, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_NEAREST)

            patch_stego = embed_packet_into_patch(patch, packets[idx])

            M_inv = cv2.invertAffineTransform(M)
            patch_back = cv2.warpAffine(patch_stego, M_inv, (img_gray.shape[1], img_gray.shape[0]),
                                        flags=cv2.INTER_NEAREST)
            mask_patch = np.full((PATCH_SIZE, PATCH_SIZE), 255, dtype=np.uint8)
            mask_warped = cv2.warpAffine(mask_patch, M_inv, (img_gray.shape[1], img_gray.shape[0]),
                                         flags=cv2.INTER_NEAREST)

            region = (mask_warped > 10)
            current_stego[region] = patch_back[region]

        stego_images.append(current_stego)
        print(f"  ✓ 嵌入完成")

    # [Phase 2] 提取
    print(f"\n[Phase 2] 提取验证 (搜索前 {SEARCH_RANGE} 个特征点)...")
    decoder = lt_min.LTDecoder()
    decoder.set_params(k=encoder.k, block_size=BLOCK_SIZE_FOR_LT, msg_len=len(secret_msg), base_seed=2025)

    recovered_total = 0

    for i, stego_img in enumerate(stego_images):
        sift = cv2.SIFT_create()
        kps_extract = sift.detect(stego_img, None)

        # 提取时也做边界过滤
        kps_boundary = filter_keypoints_by_boundary(kps_extract, stego_img.shape)
        kps_extract = sorted(kps_boundary, key=lambda x: -x.response)[:SEARCH_RANGE]

        print(f"\n图片 #{i + 1}:")
        print(f"  边界过滤后特征点: {len(kps_extract)} 个")

        count = 0
        crc_failed = 0

        for idx, kp in enumerate(kps_extract):
            M = get_patch_transform(kp, PATCH_SIZE)
            patch = cv2.warpAffine(stego_img, M, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_NEAREST)
            raw = extract_packet_from_patch(patch)

            try:
                pkt = lt_min.deserialize_lt_packet(raw, BLOCK_SIZE_FOR_LT)

                old_crc = decoder.packets_crc_failed
                old_cnt = decoder.packets_received
                decoder.add_packet(pkt)

                if decoder.packets_crc_failed > old_crc:
                    crc_failed += 1

                if decoder.packets_received > old_cnt:
                    count += 1
                    print(f"    ✓ 第 {idx} 个特征点: 成功解码新包 (总计: {decoder.packets_received}/{encoder.k})")

                    if decoder.is_decoded():
                        print(f"    🎉 已集齐所有包!")
                        break
            except:
                pass

        print(f"  成功解码: {count} 包, CRC失败: {crc_failed}")
        recovered_total += count

    print(f"\n{'=' * 50}")
    print(f"总回收: {recovered_total}/{PACKETS_PER_IMG * NUM_IMAGES}")
    print(f"解码进度: {decoder.packets_received}/{encoder.k}")

    if decoder.is_decoded():
        result = decoder.reconstruct(verify_crc=True)
        print(f"🎉 解码成功!")
        print(f"恢复内容: {result[:50]}...")
    else:
        print(f"⚠️ 未完全解码 (需要 {encoder.k} 个包，已收到 {decoder.packets_received})")


if __name__ == "__main__":
    main()