# -*- coding: utf-8 -*-
import os
import sys
import math
import struct
import shutil
import glob
from typing import List, Dict, Any

import numpy as np
import cv2
import matplotlib.pyplot as plt
import bchlib

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
#  -- 配置中心 --
# ==============================================================================
lt_module_path = r"D:\paper\fountain\lt_min.py"
input_image_directory = r"D:\paper\tools\extracted_i_only\I"
output_directory = r"D:\paper\tools\watermarked_i_only_output"
my_watermark_text = "The final, correct version. All bugs are fixed."
TARGET_K = 4
REDUNDANCY = 3.0
PACKETS_PER_FRAME = 3
QUANTIZATION_STEP = 60  # 保持一个较高的Q值

# ==============================================================================
#  -- 代码主体部分 --
# ==============================================================================

# --- 动态导入 lt_min.py 模块 ---
try:
    module_dir = os.path.dirname(lt_module_path)
    if module_dir not in sys.path: sys.path.append(module_dir)
    from lt_min import LTEncoder, LTDecoder, LTPacket

    print(f"成功从 '{module_dir}' 导入 lt_min 模块。")
except ImportError:
    print(f"错误: 无法在路径 '{lt_module_path}' 找到或导入 lt_min.py。");
    sys.exit(1)

# --- 辅助函数与常量 ---
PATCH_SIZE = 64
HEADER_SIZE = 5

BCH_POLYNOMIAL = 8219
BCH_BITS_CORRECTION = 16

bch = None


def initialize_bch(packet_size_bytes: int):
    global bch
    if bch is None:
        bch = bchlib.BCH(BCH_BITS_CORRECTION, BCH_POLYNOMIAL)
        print(f"BCH码初始化: 每个包将增加 {bch.ecc_bits} 个校验比特，可纠正最多 {BCH_BITS_CORRECTION} 个错误。")


def bytes_to_binary_string(data: bytes) -> str: return ''.join(format(byte, '08b') for byte in data)


def binary_string_to_bytes(s: str) -> bytes:
    if len(s) % 8 != 0: s = '0' * (8 - len(s) % 8) + s
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')


def get_coefficient_path(path_length):
    path = []
    for i in range(8, 48):
        for j in range(8, 48):
            if len(path) < path_length: path.append((i, j))
    if len(path) < path_length: raise ValueError(f"PATCH_SIZE={PATCH_SIZE} 无法提供 {path_length} 的路径!")
    return path


def calculate_block_size_for_k(message_size_bytes: int, target_k: int) -> int:
    if message_size_bytes == 0: return 1
    return math.ceil(message_size_bytes / target_k)


# --- 核心嵌入与提取函数 ---
def embed_on_image_sequence(image_paths: List[str], output_dir: str, encoder: LTEncoder):
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    physical_packet_size_bytes = HEADER_SIZE + encoder.block_size
    initialize_bch(physical_packet_size_bytes)
    bits_per_packet_with_bch = (physical_packet_size_bytes * 8) + bch.ecc_bits

    print(f"源消息分为 {encoder.k} 个块。")
    print(f"物理包大小: {physical_packet_size_bytes} 字节。")
    print(f"BCH编码后总大小: {bits_per_packet_with_bch} 比特。")
    print(f"策略: 在【每张】可用图片上嵌入 {PACKETS_PER_FRAME} 个包。")

    keypoints_per_frame: Dict[str, Any] = {}
    watermarked_frame_paths: List[str] = []
    total_packets_embedded = 0

    for frame_idx, image_path in enumerate(image_paths):
        print(f"\r处理图片 {frame_idx + 1}/{len(image_paths)}: {os.path.basename(image_path)}...", end="")
        original_img = cv2.imread(image_path)
        if original_img is None: continue

        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, _ = sift.detectAndCompute(gray_img, None)
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)

        selected_keypoints = []
        used_locations = set()
        for kp in keypoints:
            if len(selected_keypoints) >= PACKETS_PER_FRAME: break
            location = (int(kp.pt[0]), int(kp.pt[1]))
            if location not in used_locations:
                selected_keypoints.append(kp)
                used_locations.add(location)

        if len(selected_keypoints) < PACKETS_PER_FRAME:
            # print(f"\n警告: 图片 {os.path.basename(image_path)} 的唯一特征点不足。需要 {PACKETS_PER_FRAME}, 找到 {len(selected_keypoints)}。跳过。")
            continue

        watermarked_img = original_img.copy()
        packets_for_this_frame = [encoder.next_packet() for _ in range(PACKETS_PER_FRAME)]

        for i, pkt in enumerate(packets_for_this_frame):
            kp = selected_keypoints[i]
            if pkt.sys_idx is not None:
                flag, value = b'\x00', struct.pack('!I', pkt.sys_idx)
            else:
                flag, value = b'\x01', struct.pack('!I', pkt.seed)
            physical_payload_bytes = flag + value + pkt.payload

            ecc = bch.encode(physical_payload_bytes)
            packet_with_ecc = physical_payload_bytes + ecc
            bits_chunk = bytes_to_binary_string(packet_with_ecc)

            x, y = int(kp.pt[0]), int(kp.pt[1])
            patch_radius = int(kp.size * 2)
            h, w = gray_img.shape
            top, bottom = max(0, y - patch_radius), min(h, y + patch_radius)
            left, right = max(0, x - patch_radius), min(w, x + patch_radius)
            patch = watermarked_img[top:bottom, left:right, 0].copy()
            if patch.shape[0] < PATCH_SIZE or patch.shape[1] < PATCH_SIZE: continue
            normalized = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
            dct_patch = cv2.dct(normalized.astype(np.float32))
            coeff_path = get_coefficient_path(len(bits_chunk))
            for bit_idx in range(len(bits_chunk)):
                bit_to_embed = int(bits_chunk[bit_idx])
                coeff_pos = coeff_path[bit_idx]
                q_val = round(dct_patch[coeff_pos] / QUANTIZATION_STEP)
                if bit_to_embed == 0:
                    if q_val % 2 != 0: q_val -= 1
                else:
                    if q_val % 2 == 0: q_val += 1
                dct_patch[coeff_pos] = q_val * QUANTIZATION_STEP
            modified_norm = cv2.idct(dct_patch)
            modified_patch = cv2.resize(modified_norm, (patch.shape[1], patch.shape[0]))
            watermarked_img[top:bottom, left:right, 0] = np.clip(modified_patch, 0, 255).astype(np.uint8)

        output_filename = os.path.join(output_dir, f"wm_{os.path.basename(image_path)}")
        cv2.imwrite(output_filename, watermarked_img)
        watermarked_frame_paths.append(output_filename)
        keypoints_per_frame[output_filename] = selected_keypoints
        total_packets_embedded += len(packets_for_this_frame)

    print(f"\n嵌入流程完成。总共在 {len(watermarked_frame_paths)} 张图片中嵌入了 {total_packets_embedded} 个包。")
    return watermarked_frame_paths, keypoints_per_frame


def extract_watermark_from_frames(decoder: LTDecoder, watermarked_paths: List[str], kps_info: Dict[str, Any],
                                  initial_k: int, initial_block_size: int, initial_msg_len: int):
    print("\n--- 开始从图片序列中提取并解码 ---")

    physical_packet_size_bytes = HEADER_SIZE + initial_block_size
    initialize_bch(physical_packet_size_bytes)
    bits_per_packet_with_bch = (physical_packet_size_bytes * 8) + bch.ecc_bits

    total_corrected_errors = 0
    total_failed_packets = 0

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 核心修正：将 extracted_bits_str 移到循环外 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    all_extracted_bits = ""
    for frame_path in watermarked_paths:
        print(f"\r正在提取: {os.path.basename(frame_path)}", end="")
        watermarked_img = cv2.imread(frame_path)
        if watermarked_img is None or frame_path not in kps_info: continue
        keypoints_for_this_frame = kps_info[frame_path]

        for kp in keypoints_for_this_frame:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            patch_radius = int(kp.size * 2)
            h, w, _ = watermarked_img.shape
            top, bottom = max(0, y - patch_radius), min(h, y + patch_radius)
            left, right = max(0, x - patch_radius), min(w, x + patch_radius)
            patch = watermarked_img[top:bottom, left:right, 0]
            if patch.shape[0] < PATCH_SIZE or patch.shape[1] < PATCH_SIZE: continue

            normalized = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
            dct_patch = cv2.dct(normalized.astype(np.float32))

            coeff_path = get_coefficient_path(bits_per_packet_with_bch)
            for i in range(bits_per_packet_with_bch):
                coeff_pos = coeff_path[i]
                q_val = round(dct_patch[coeff_pos] / QUANTIZATION_STEP)
                all_extracted_bits += '1' if q_val % 2 != 0 else '0'

    print(f"\n\n提取流程完成，总共累积了 {len(all_extracted_bits)} 个比特。")
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    num_packets_to_process = len(all_extracted_bits) // bits_per_packet_with_bch
    print(f"--- 开始解码，尝试从比特流中重建 {num_packets_to_process} 个物理包... ---")

    for i in range(num_packets_to_process):
        start = i * bits_per_packet_with_bch
        end = start + bits_per_packet_with_bch
        extracted_bits_chunk = all_extracted_bits[start:end]

        try:
            packet_with_ecc_bytes = binary_string_to_bytes(extracted_bits_chunk)
            data = bytearray(packet_with_ecc_bytes[:-bch.ecc_bytes])
            ecc = packet_with_ecc_bytes[-bch.ecc_bytes:]
            errors = bch.decode_inplace(data, ecc)
            total_corrected_errors += errors
            if errors > 0:
                print(f"\n[BCH] 在第 {i + 1} 个包中纠正了 {errors} 个比特错误。", end="")
            physical_packet_bytes = data
        except Exception:
            total_failed_packets += 1
            continue

        flag, value_bytes, payload = physical_packet_bytes[0:1], physical_packet_bytes[1:5], physical_packet_bytes[5:]
        value, = struct.unpack('!I', value_bytes)
        if flag == b'\x00':
            pkt = LTPacket(k=initial_k, block_size=initial_block_size, msg_len=initial_msg_len, seed=None,
                           sys_idx=value, payload=payload)
        else:
            pkt = LTPacket(k=initial_k, block_size=initial_block_size, msg_len=initial_msg_len, seed=value,
                           sys_idx=None, payload=payload)

        decoder.add_packet(pkt)
        if decoder.is_decoded():
            print(f"\n\n好消息！在处理完第 {i + 1} 个包后解码成功！")
            print(f"BCH总计纠正了 {total_corrected_errors} 个比特错误。")
            print(f"BCH解码失败 (错误过多) 的包数量: {total_failed_packets}")
            return decoder.reconstruct()

    print(f"\n\n坏消息... 所有包都处理完后，仍未能解码。")
    print(f"BCH总计纠正了 {total_corrected_errors} 个比特错误。")
    print(f"BCH解码失败 (错误过多) 的包数量: {total_failed_packets}")
    return None


# --- 主程序入口 ---
if __name__ == '__main__':
    # ... (与上一版完全相同) ...
    if not os.path.isdir(input_image_directory):
        print(f"错误: 输入文件夹不存在: '{input_image_directory}'");
        sys.exit(1)

    image_files = sorted(glob.glob(os.path.join(input_image_directory, '*.jpg')))
    if not image_files:
        print(f"错误: 在文件夹 '{input_image_directory}' 中没有找到任何 .jpg 图片。");
        sys.exit(1)

    print(f"在输入文件夹中找到 {len(image_files)} 张图片待处理。")

    message_bytes = my_watermark_text.encode('utf-8')
    dynamic_block_size = calculate_block_size_for_k(len(message_bytes), TARGET_K)
    print(f"信息大小 {len(message_bytes)} 字节, 动态计算 block_size = {dynamic_block_size}, 以确保 k = {TARGET_K}。")

    encoder = LTEncoder(message=message_bytes, block_size=dynamic_block_size, base_seed=1234)

    print("\n--- 开始在图片序列上进行【BCH保护】嵌入 ---")
    watermarked_paths, kps_info = embed_on_image_sequence(
        image_files, output_directory, encoder)

    if not watermarked_paths:
        print("\n没有生成任何带水印的图片，流程终止。")
    else:
        decoder = LTDecoder()
        reconstructed_bytes = extract_watermark_from_frames(
            decoder, watermarked_paths, kps_info, encoder.k, encoder.block_size, len(encoder.msg))

        if reconstructed_bytes:
            reconstructed_text = reconstructed_bytes.decode('utf-8', errors='ignore')
            print("\n" + "=" * 20 + " 最终结果 " + "=" * 20)
            print(f"原始消息: '{my_watermark_text}'")
            print(f"重构消息: '{reconstructed_text}'")
            if my_watermark_text == reconstructed_text:
                print("\n[SUCCESS] 验证成功！")
            else:
                print("\n[FAILURE] 验证失败！")
        else:
            print("\n" + "=" * 20 + " 最终结果 " + "=" * 20)
            print("[FAILURE] 解码失败。")

        # ... (可视化模块与上一版完全相同) ...
        print("\n--- 正在为所有成功嵌入的图片生成可视化对比图 ---")
        visualization_dir = os.path.join(output_directory, "visualization_output")
        if os.path.exists(visualization_dir): shutil.rmtree(visualization_dir)
        os.makedirs(visualization_dir, exist_ok=True)

        MARKER_RADIUS = 50;
        MARKER_COLOR = (0, 0, 255);
        MARKER_THICKNESS = 2

        for watermarked_path in watermarked_paths:
            if watermarked_path not in kps_info: continue
            original_basename = os.path.basename(watermarked_path)[3:]
            original_image_path = os.path.join(input_image_directory, original_basename)
            keypoints = kps_info[watermarked_path]
            original_image = cv2.imread(original_image_path)
            watermarked_image = cv2.imread(watermarked_path)
            visualization_img = watermarked_image.copy()
            for kp in keypoints:
                center = (int(kp.pt[0]), int(kp.pt[1]))
                cv2.circle(visualization_img, center, MARKER_RADIUS, MARKER_COLOR, MARKER_THICKNESS)
                cv2.line(visualization_img, (center[0] - MARKER_RADIUS, center[1]),
                         (center[0] + MARKER_RADIUS, center[1]), MARKER_COLOR, MARKER_THICKNESS)
                cv2.line(visualization_img, (center[0], center[1] - MARKER_RADIUS),
                         (center[0], center[1] + MARKER_RADIUS), MARKER_COLOR, MARKER_THICKNESS)
            h, w, _ = original_image.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(original_image, '1. 原始图片', (50, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(watermarked_image, '2. 嵌入水印后', (50, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(visualization_img, '3. 水印位置标记', (50, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
            combined_image = np.hstack((original_image, watermarked_image, visualization_img))
            viz_output_path = os.path.join(visualization_dir, f"viz_{original_basename}")
            cv2.imwrite(viz_output_path, combined_image)
        print(f"\n可视化流程完成！所有对比图已保存至:\n{os.path.abspath(visualization_dir)}")