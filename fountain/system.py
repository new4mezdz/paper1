# -*- coding: utf-8 -*-
"""
喷泉码水印系统 - 完整版 (含几何攻击测试)
- 心跳包机制（不需要预知 k）
- 批量嵌入文件夹下所有图片
- 手动选择图片进行提取
- [新增] 提取前可选多种攻击（JPEG、平移、缩放、裁剪、旋转）
- [修复] 使用灰度图处理，避免YCrCb转换误差
"""
import cv2
import numpy as np
import os
import struct
import zlib
from glob import glob
import lt_min
import qimtest

# =================配置区域=================
INPUT_FOLDER = r"D:\paper data\output_3\10"  # 输入图片文件夹
OUTPUT_FOLDER = r"D:\paper1\fountain\extracted_i_only\I"  # 输出 stego 图片

PAYLOAD_SIZE = 31  # 包大小（心跳包和数据包统一）
BLOCK_SIZE_FOR_LT = 23  # LT 编码 payload
PATCH_SIZE = 128  # 补丁大小
QIM_STEP = 200  # QIM 步长

PACKETS_PER_IMG = 15  # 每张图嵌入的包数量（含1个心跳包 + 14个数据包）
SEARCH_RANGE = 300  # 提取时搜索的特征点数量
BASE_SEED = 2025  # 随机种子

# 心跳包 magic header
HEARTBEAT_MAGIC = 0xDEADBEEF


# ==========================================
# 攻击工具
# ==========================================
def attack_jpeg(img, quality):
    """
    对图像进行 JPEG 压缩攻击
    img: BGR 或 Gray 图像
    quality: 1-100 (越低越狠)
    返回: 压缩后的灰度图
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 0)  # 强制返回灰度图


def attack_translate(img, tx, ty):
    """
    平移攻击
    tx, ty: 平移像素数（可为负）
    """
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def attack_scale(img, scale_factor):
    """
    缩放攻击
    scale_factor: 缩放比例（如 0.8 缩小, 1.2 放大）
    """
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # 再缩放回原尺寸
    return cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)


def attack_crop(img, crop_ratio):
    """
    裁剪攻击（中心裁剪后放大回原尺寸）
    crop_ratio: 保留比例（如 0.9 表示裁剪掉 10%）
    """
    h, w = img.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    y1, x1 = (h - ch) // 2, (w - cw) // 2
    cropped = img[y1:y1 + ch, x1:x1 + cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def attack_rotate(img, angle):
    """
    旋转攻击（保留完整图片，扩大画布）
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    # 计算旋转后需要的新尺寸
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # 调整平移，让图片居中
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(img, M, (new_w, new_h), borderValue=(0, 0, 0))


# ==========================================
# 心跳包处理
# ==========================================
def create_heartbeat_packet(k, block_size, msg_len, base_seed, msg_crc):
    """
    创建心跳包（31字节）
    格式: magic(4) + k(4) + block_size(4) + msg_len(4) + base_seed(4) + msg_crc(4) + crc(4) + padding(3)
    """
    data = struct.pack('>I', HEARTBEAT_MAGIC)  # 4 bytes
    data += struct.pack('>I', k)  # 4 bytes
    data += struct.pack('>I', block_size)  # 4 bytes
    data += struct.pack('>I', msg_len)  # 4 bytes
    data += struct.pack('>I', base_seed)  # 4 bytes
    data += struct.pack('>I', msg_crc)  # 4 bytes
    # 计算 CRC
    crc = zlib.crc32(data) & 0xFFFFFFFF
    data += struct.pack('>I', crc)  # 4 bytes
    data += b'\x00' * 3  # 3 bytes padding
    return data  # 总共 31 bytes


def parse_heartbeat_packet(data):
    """
    解析心跳包
    返回: (k, block_size, msg_len, base_seed, msg_crc) 或 None
    """
    if len(data) != PAYLOAD_SIZE:
        return None

    magic = struct.unpack('>I', data[0:4])[0]
    if magic != HEARTBEAT_MAGIC:
        return None

    k = struct.unpack('>I', data[4:8])[0]
    block_size = struct.unpack('>I', data[8:12])[0]
    msg_len = struct.unpack('>I', data[12:16])[0]
    base_seed = struct.unpack('>I', data[16:20])[0]
    msg_crc = struct.unpack('>I', data[20:24])[0]
    stored_crc = struct.unpack('>I', data[24:28])[0]

    # 验证 CRC
    expected_crc = zlib.crc32(data[0:24]) & 0xFFFFFFFF
    if expected_crc != stored_crc:
        return None

    # 合理性检查
    if k <= 0 or k > 10000 or block_size <= 0 or block_size > 1000:
        return None

    return k, block_size, msg_len, base_seed, msg_crc


# ==========================================
# 基础函数
# ==========================================
def get_patch_transform(kp, output_size):
    """量化角度，提高鲁棒性"""
    x, y = kp.pt
    angle = kp.angle

    M = cv2.getRotationMatrix2D((x, y), -angle, 1.0)
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


def filter_keypoints_by_boundary(kps_all, img_shape):
    """过滤：只保留离边界足够远的点"""
    h, w = img_shape
    half = PATCH_SIZE // 2
    margin = half + 10

    kp_filtered = []
    for kp in kps_all:
        x, y = kp.pt
        if margin < x < w - margin and margin < y < h - margin:
            kp_filtered.append(kp)

    return kp_filtered


def filter_keypoints_by_distance(kps, min_dist=PATCH_SIZE):
    """过滤掉距离太近的点"""
    selected = []
    for kp in kps:
        x, y = kp.pt
        too_close = False
        for sel in selected:
            sx, sy = sel.pt
            if np.sqrt((x - sx) ** 2 + (y - sy) ** 2) < min_dist:
                too_close = True
                break
        if not too_close:
            selected.append(kp)
    return selected


def filter_stable_keypoints(img_gray, candidates):
    """数据完整性过滤：只保留嵌入后仍能被SIFT找到且能正确提取数据的点"""
    h, w = img_gray.shape
    dummy_data = b'\xAA' * PAYLOAD_SIZE

    survivors = []
    sift = cv2.SIFT_create()

    for kp in candidates:
        try:
            M = get_patch_transform(kp, PATCH_SIZE)
            patch = cv2.warpAffine(img_gray, M, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_NEAREST)
            patch_stego = embed_packet_into_patch(patch, dummy_data)

            temp_img = img_gray.copy()
            M_inv = cv2.invertAffineTransform(M)
            patch_back = cv2.warpAffine(patch_stego, M_inv, (w, h), flags=cv2.INTER_NEAREST)
            mask = np.full((PATCH_SIZE, PATCH_SIZE), 255, dtype=np.uint8)
            mask_warped = cv2.warpAffine(mask, M_inv, (w, h), flags=cv2.INTER_NEAREST)
            region = (mask_warped > 10)
            temp_img[region] = patch_back[region]

            # 新增：检查嵌入后SIFT是否还能找到这个点（位置和角度都要接近）
            kps_after = sift.detect(temp_img, None)
            found = False
            for new_kp in kps_after:
                dist = np.sqrt((kp.pt[0] - new_kp.pt[0]) ** 2 + (kp.pt[1] - new_kp.pt[1]) ** 2)
                angle_diff = abs(kp.angle - new_kp.angle)
                angle_diff = min(angle_diff, 360 - angle_diff)
                if dist < 3 and angle_diff < 5:
                    found = True
                    break

            if not found:
                continue

            # 用新检测到的点来提取验证
            patch_verify = cv2.warpAffine(temp_img, M, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_NEAREST)
            extracted = extract_packet_from_patch(patch_verify)

            if extracted == dummy_data:
                survivors.append(kp)
        except:
            continue


    return survivors

def embed_into_image(img_bgr, packets):
    """
    将多个包嵌入到单张图片（灰度处理）
    packets: 包列表（第一个应该是心跳包）
    返回: stego 灰度图片, 实际嵌入的包数
    """
    # 直接转灰度，不用 YCrCb
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    # 检测特征点
    sift = cv2.SIFT_create()
    kps_all = sift.detect(img_gray, None)

    # 边界过滤
    kps_boundary = filter_keypoints_by_boundary(kps_all, img_gray.shape)

    # 按响应排序 + 距离过滤
    candidates = sorted(kps_boundary, key=lambda x: -x.response)
    candidates = filter_keypoints_by_distance(candidates)[:100]

    # 数据完整性过滤
    stable_kps = filter_stable_keypoints(img_gray, candidates)

    # 选择最终使用的点
    target_kps = stable_kps[:len(packets)]

    if len(target_kps) < len(packets):
        print(f"    ⚠️ 只有 {len(target_kps)} 个稳定点，需要 {len(packets)} 个")

    # 嵌入
    current_stego = img_gray.copy()
    embedded_count = 0

    for idx, kp in enumerate(target_kps):
        if idx >= len(packets):
            break

        M = get_patch_transform(kp, PATCH_SIZE)
        patch = cv2.warpAffine(current_stego, M, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_NEAREST)
        patch_stego = embed_packet_into_patch(patch, packets[idx])

        M_inv = cv2.invertAffineTransform(M)
        patch_back = cv2.warpAffine(patch_stego, M_inv, (w, h), flags=cv2.INTER_NEAREST)
        mask_patch = np.full((PATCH_SIZE, PATCH_SIZE), 255, dtype=np.uint8)
        mask_warped = cv2.warpAffine(mask_patch, M_inv, (w, h), flags=cv2.INTER_NEAREST)

        region = (mask_warped > 10)
        current_stego[region] = patch_back[region]
        embedded_count += 1
    print(f"    [调试] 稳定点: {len(stable_kps)}, 实际嵌入: {embedded_count}")

    # 嵌入后立即验证
    sift_verify = cv2.SIFT_create()
    kps_after = sift_verify.detect(current_stego, None)
    kps_after = filter_keypoints_by_boundary(kps_after, (h, w))
    kps_after = sorted(kps_after, key=lambda x: -x.response)[:SEARCH_RANGE]

    verify_ok = 0
    for kp in kps_after:
        M = get_patch_transform(kp, PATCH_SIZE)
        patch = cv2.warpAffine(current_stego, M, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_NEAREST)
        raw = extract_packet_from_patch(patch)
        hb = parse_heartbeat_packet(raw)
        if hb is not None:
            verify_ok += 1
            continue
        try:
            pkt = lt_min.deserialize_lt_packet(raw, BLOCK_SIZE_FOR_LT)
            if pkt.verify_crc():
                verify_ok += 1
        except:
            pass
    print(f"    [验证] 嵌入后立即提取: {verify_ok}/{embedded_count}")
    # 直接返回灰度图
    return current_stego, embedded_count


def extract_from_image(img_gray):
    """从灰度图中提取水印"""
    h, w = img_gray.shape
    sift = cv2.SIFT_create()
    kps_all = sift.detect(img_gray, None)
    kps_boundary = filter_keypoints_by_boundary(kps_all, (h, w))
    kps_sorted = sorted(kps_boundary, key=lambda x: -x.response)[:SEARCH_RANGE]

    heartbeats = []
    data_packets = []

    for kp in kps_sorted:
        M = get_patch_transform(kp, PATCH_SIZE)
        patch = cv2.warpAffine(img_gray, M, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_NEAREST)
        raw = extract_packet_from_patch(patch)

        # 尝试解析为心跳包
        hb = parse_heartbeat_packet(raw)
        if hb is not None:
            heartbeats.append(hb)
            continue

        # 尝试解析为数据包
        try:
            pkt = lt_min.deserialize_lt_packet(raw, BLOCK_SIZE_FOR_LT)
            data_packets.append(pkt)
        except:
            pass

    return heartbeats, data_packets


# ==========================================
# 嵌入模式
# ==========================================
def main_embed():
    """嵌入模式：处理文件夹下所有图片"""
    print("\n" + "=" * 60)
    print("  嵌入模式")
    print("=" * 60)

    # 获取秘密消息
    secret_msg = input("请输入要嵌入的秘密消息: ").encode('utf-8')
    if not secret_msg:
        secret_msg = b"Hajimi-sama's Robust System - Default Message"

    print(f"\n消息长度: {len(secret_msg)} 字节")

    # 创建编码器
    encoder = lt_min.LTEncoder(secret_msg, block_size=BLOCK_SIZE_FOR_LT, base_seed=BASE_SEED)
    print(f"需要 k={encoder.k} 个源块")

    # 创建心跳包
    heartbeat = create_heartbeat_packet(
        k=encoder.k,
        block_size=BLOCK_SIZE_FOR_LT,
        msg_len=len(secret_msg),
        base_seed=BASE_SEED,
        msg_crc=encoder.msg_crc
    )

    # 获取所有输入图片
    img_files = glob(os.path.join(INPUT_FOLDER, "*.png"))
    img_files += glob(os.path.join(INPUT_FOLDER, "*.jpg"))
    img_files = sorted(img_files)

    if not img_files:
        print(f"错误: 在 {INPUT_FOLDER} 找不到图片")
        return

    print(f"找到 {len(img_files)} 张图片")

    # 创建输出文件夹
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 计算每张图需要嵌入多少包
    data_packets_per_img = PACKETS_PER_IMG - 1  # 减去心跳包

    total_embedded = 0
    processed_images = 0

    for i, img_path in enumerate(img_files):
        print(f"\n[{i + 1}/{len(img_files)}] 处理: {os.path.basename(img_path)}")

        # 读取图片
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print("  跳过: 无法读取")
            continue

        # 生成数据包
        data_packets = [lt_min.serialize_lt_packet(encoder.next_packet())
                        for _ in range(data_packets_per_img)]

        # 组合包列表：心跳包 + 数据包
        all_packets = []
        for j, pkt in enumerate(data_packets):
            if j % 4 == 0:  # 每4个数据包前插入1个心跳包
                all_packets.append(heartbeat)
            all_packets.append(pkt)

        # 确保不超过 PACKETS_PER_IMG
        all_packets = all_packets[:PACKETS_PER_IMG]

        # 嵌入
        stego_img, embedded_count = embed_into_image(img_bgr, all_packets)

        print(f"  嵌入: {embedded_count} 包 (1 心跳 + {embedded_count - 1} 数据)")

        # 保存（灰度图直接保存为 PNG）
        output_path = os.path.join(OUTPUT_FOLDER, f"stego_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, stego_img)
        print(f"  保存: {output_path}")

        total_embedded += embedded_count
        processed_images += 1

    print("\n" + "=" * 60)
    print(f"嵌入完成!")
    print(f"处理图片: {processed_images} 张")
    print(f"总嵌入包数: {total_embedded}")
    print(f"输出目录: {OUTPUT_FOLDER}")
    print("=" * 60)


# ==========================================
# 提取模式 (含攻击)
# ==========================================
def main_extract():
    """提取模式：让用户选择图片"""
    print("\n" + "=" * 60)
    print("  提取模式 (支持攻击测试)")
    print("=" * 60)

    # 获取所有 stego 图片
    stego_files = glob(os.path.join(OUTPUT_FOLDER, "*.png"))
    stego_files += glob(os.path.join(OUTPUT_FOLDER, "*.jpg"))
    stego_files += glob(os.path.join(OUTPUT_FOLDER, "*.jpeg"))
    stego_files = sorted(stego_files)

    if not stego_files:
        print(f"错误: 在 {OUTPUT_FOLDER} 找不到 stego 图片")
        return

    # 显示可选图片
    print(f"\n找到 {len(stego_files)} 张 stego 图片:\n")
    for i, f in enumerate(stego_files):
        print(f"  [{i + 1}] {os.path.basename(f)}")

    # 用户选择
    print(f"\n请输入要提取的图片编号（用逗号分隔，或输入 'all' 选择全部）:")
    choice = input("> ").strip()

    if choice.lower() == 'all':
        selected_files = stego_files
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected_files = [stego_files[i] for i in indices if 0 <= i < len(stego_files)]
        except:
            print("输入无效")
            return

    if not selected_files:
        print("没有选择任何图片")
        return

    # === 攻击选择菜单 ===
    print("\n" + "-" * 40)
    print("选择攻击类型（可组合，用逗号分隔）:")
    print("  0 = 不攻击")
    print("  1 = JPEG 压缩")
    print("  2 = 平移")
    print("  3 = 缩放")
    print("  4 = 裁剪")
    print("  5 = 旋转")
    attack_choice = input("> ").strip()

    attacks = {}
    if attack_choice and attack_choice != '0':
        for a in attack_choice.split(','):
            a = a.strip()
            if a == '1':
                q = input("  JPEG 质量 (1-100): ").strip()
                attacks['jpeg'] = int(q) if q.isdigit() else 60
            elif a == '2':
                tx = input("  平移 X 像素: ").strip()
                ty = input("  平移 Y 像素: ").strip()
                attacks['translate'] = (int(tx), int(ty))
            elif a == '3':
                s = input("  缩放比例 (如 0.9): ").strip()
                attacks['scale'] = float(s)
            elif a == '4':
                r = input("  裁剪保留比例 (如 0.9): ").strip()
                attacks['crop'] = float(r)
            elif a == '5':
                ang = input("  旋转角度 (度): ").strip()
                attacks['rotate'] = float(ang)

    if attacks:
        print(f"⚠️ 已开启攻击: {list(attacks.keys())}")
    else:
        print("✅ 无损提取模式")
    print("-" * 40 + "\n")

    print(f"选择了 {len(selected_files)} 张图片，开始提取...")

    # 收集所有包
    all_heartbeats = []
    all_data_packets = []

    for img_path in selected_files:
        print(f"\n提取: {os.path.basename(img_path)}")

        # 读取图片（直接读灰度）
        img_to_process = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_to_process is None:
            print("  跳过: 无法读取")
            continue

        # === 应用攻击 ===
        if 'jpeg' in attacks:
            img_to_process = attack_jpeg(img_to_process, attacks['jpeg'])
            print(f"  [攻击] JPEG Q={attacks['jpeg']}")
        if 'translate' in attacks:
            tx, ty = attacks['translate']
            img_to_process = attack_translate(img_to_process, tx, ty)
            print(f"  [攻击] 平移 ({tx}, {ty})")
        if 'scale' in attacks:
            img_to_process = attack_scale(img_to_process, attacks['scale'])
            print(f"  [攻击] 缩放 {attacks['scale']}")
        if 'crop' in attacks:
            img_to_process = attack_crop(img_to_process, attacks['crop'])
            print(f"  [攻击] 裁剪 {attacks['crop']}")
        if 'rotate' in attacks:
            img_to_process = attack_rotate(img_to_process, attacks['rotate'])
            print(f"  [攻击] 旋转 {attacks['rotate']}°")

        # 送入提取器
        heartbeats, data_packets = extract_from_image(img_to_process)

        print(f"  心跳包: {len(heartbeats)} 个, 候选数据包: {len(data_packets)} 个")

        all_heartbeats.extend(heartbeats)
        all_data_packets.extend(data_packets)

    print(f"\n总计: 心跳包 {len(all_heartbeats)} 个, 数据包 {len(all_data_packets)} 个")

    # 从心跳包获取参数
    if not all_heartbeats:
        print("\n❌ 未找到心跳包，无法解码")
        return

    # 使用第一个有效心跳包的参数
    k, block_size, msg_len, base_seed, msg_crc = all_heartbeats[0]
    print(f"\n从心跳包获取参数:")
    print(f"  k={k}, block_size={block_size}, msg_len={msg_len}")
    print(f"  base_seed=0x{base_seed:08X}, msg_crc=0x{msg_crc:08X}")

    # 创建解码器
    decoder = lt_min.LTDecoder()
    decoder.set_params(k=k, block_size=block_size, msg_len=msg_len, base_seed=base_seed, msg_crc=msg_crc)

    # 添加数据包
    for pkt in all_data_packets:
        decoder.add_packet(pkt)

    print(f"\n解码进度: {decoder.packets_received}/{k}")
    print(f"CRC 失败: {decoder.packets_crc_failed}")
    print(f"有效数据包: {decoder.packets_received}")
    print(f"重复包: {decoder.packets_duplicate}")

    # 尝试解码
    if decoder.is_decoded():
        try:
            result = decoder.reconstruct(verify_crc=True)
            print(f"\n🎉 解码成功!")
            print(f"\n恢复的消息 ({len(result)} 字节):")
            print("-" * 40)
            try:
                print(result.decode('utf-8'))
            except:
                print(result)
            print("-" * 40)
        except Exception as e:
            print(f"\n❌ 解码失败: {e}")
    else:
        print(f"\n⚠️ 包数不足，需要 {k} 个，已收到 {decoder.packets_received} 个")
        print("请选择更多图片重试")


# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  喷泉码水印系统 (整合攻击测试版)")
    print("=" * 60)
    print("\n请选择模式:")
    print("  [1] 嵌入 - 将消息嵌入到图片")
    print("  [2] 提取 - 从图片中恢复消息")

    choice = input("\n请选择 (1/2): ").strip()

    if choice == '1':
        main_embed()
    elif choice == '2':
        main_extract()
    else:
        print("无效选择")