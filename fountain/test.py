# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random
import os
import lt_min  # 必须确保 lt_min.py 在同目录下
import qimtest  # 必须确保 qimtest.py 在同目录下

# =================配置区域=================
IMG_PATH = r"D:\paper data\stego_images\I_pts_364.png"

# 分布式策略配置
TOTAL_IMAGES = 10  # 总共生成几张图 (分布式存储节点数)
SELECT_IMAGES = 5  # 最终选取几张来恢复 (模拟丢失50%的图片)
PACKETS_PER_IMG = 9  # 每张图携带的数据包数量 (不含Meta包)

# 攻击与鲁棒性配置
JPEG_QUALITY = 70 # 攻击强度 (100=无损, 60=强压缩, <50=毁灭性)
QIM_STEP = 100  # QIM步长 (建议: Q=60时设80-100; Q=70时设60-80)
BLOCK_SIZE = 32  # 喷泉码单块大小 (为了塞进图片，设小一点)


# ==========================================

def extract_raw_bytes(img, step, max_bytes):
    """
    辅助函数：从图片暴力提取比特流，不关心包结构
    """
    h, w = img.shape
    img_float = img.astype(float)
    extracted_bits = []

    count = 0
    total_bits_needed = max_bytes * 8

    # 按照嵌入顺序遍历 (这里只做简单的从上到下)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            if count >= total_bits_needed:
                break
            block = img_float[y:y + 8, x:x + 8]
            dct_block = cv2.dct(block)
            coeff = dct_block[4, 3]  # 和 qimtest 保持一致的嵌入位置
            bit = qimtest.qim_extract_scalar(coeff, step)
            extracted_bits.append(bit)
            count += 1

    extracted_bits = np.array(extracted_bits, dtype=np.uint8)
    return qimtest.bits_to_bytes(extracted_bits)


if __name__ == "__main__":
    print("=" * 60)
    print(" >>> 哈吉米sama的分布式隐写存储模拟 (最终修正版) <<<")
    print(f" 载体: {os.path.basename(IMG_PATH)}")
    print(f" 策略: 生成 {TOTAL_IMAGES} 张图 -> 选取 {SELECT_IMAGES} 张 -> JPEG Q={JPEG_QUALITY}")
    print("=" * 60)

    # ------------------------------------------------
    # 0. 检查环境
    # ------------------------------------------------
    if not os.path.exists(IMG_PATH):
        print(f"[错误] 找不到图片: {IMG_PATH}")
        # 生成随机噪点图兜底
        base_img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    else:
        base_img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
        # 裁剪为8的倍数
        if base_img is None:
            print("[错误] 图片读取失败，请检查路径。生成随机图代替。")
            base_img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

        h, w = base_img.shape
        base_img = base_img[:h // 8 * 8, :w // 8 * 8]
        print(f"[载体] 读取成功，尺寸: {base_img.shape}")

    # ------------------------------------------------
    # 1. 准备秘密数据 (模拟一段机密文本)
    # ------------------------------------------------
    secret_text = "哈吉米sama的绝密论文数据！这里存放着关于喷泉码和分布式存储的核心机密..."
    secret_data = secret_text.encode('utf-8')

    # 初始化喷泉码编码器
    encoder = lt_min.LTEncoder(secret_data, block_size=BLOCK_SIZE, base_seed=2024)

    # 准备 Meta 包 (全局描述信息)
    meta_packet = encoder.get_meta_packet()
    meta_bytes = lt_min.serialize_meta_packet(meta_packet)

    print(f"\n[数据] 原始大小: {len(secret_data)} bytes")
    print(f"[数据] 喷泉码切分 k: {encoder.k} 块")

    # ------------------------------------------------
    # 2. 生成分布式图片库
    # ------------------------------------------------
    print(f"\n[嵌入] 正在生成 {TOTAL_IMAGES} 张分布式图片...")
    stego_images_db = []

    for i in range(TOTAL_IMAGES):
        # 每张图片的“载荷” = 1个Meta包 + N个数据包
        # 这样每一张图都是“自描述”的，哪怕只捡到一张，也知道文件总大小和参数
        payload_list = [meta_bytes]

        for _ in range(PACKETS_PER_IMG):
            pkt = encoder.next_packet()
            pkt_bytes = lt_min.serialize_lt_packet(pkt)
            payload_list.append(pkt_bytes)

        # 调用 qimtest 进行嵌入
        # 注意：这里使用 step=QIM_STEP 来抵抗压缩
        stego, _ = qimtest.embed_multiple_packets(base_img, payload_list, step=QIM_STEP)
        stego_images_db.append(stego)
        print(f"  -> 生成分片 #{i + 1} (含Meta + {PACKETS_PER_IMG}个数据包)")

    # ------------------------------------------------
    # 3. 模拟灾难恢复 (随机选几张 + JPEG攻击)
    # ------------------------------------------------
    print("\n" + "=" * 60)
    print(f"步骤3: 模拟传输与攻击 (仅保留 {SELECT_IMAGES} 张)")
    print("=" * 60)

    selected_indices = random.sample(range(TOTAL_IMAGES), SELECT_IMAGES)
    print(f"[选择] 接收到的图片索引: {selected_indices}")

    decoder = lt_min.LTDecoder()
    meta_initialized = False

    total_valid_packets = 0

    for idx in selected_indices:
        # A. 模拟 JPEG 攻击
        img = stego_images_db[idx]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        attacked_img = cv2.imdecode(encimg, 0)

        # B. 提取所有可能的字节
        # 估算每张图最大可能的数据量 (Meta 24 + 10个包 * (8+32) = ~424 bytes)
        # 提取 600 bytes 确保覆盖
        raw_data = extract_raw_bytes(attacked_img, step=QIM_STEP, max_bytes=600)

        # C. 解析数据流
        cursor = 0
        img_pkt_count = 0

        # C-1. 尝试解析 Meta 包 (位于头部 24 字节)
        if not meta_initialized:
            try:
                potential_meta = raw_data[0:24]
                meta_obj = lt_min.deserialize_meta_packet(potential_meta)
                decoder.set_params_from_meta(meta_obj)
                meta_initialized = True
                print(f"  [图 #{idx + 1}] ✅ Meta信息解析成功! (k={meta_obj.k})")
            except Exception as e:
                print(f"  [图 #{idx + 1}] ⚠️ Meta损坏 (JPEG噪声导致)")

        cursor += 24  # 跳过 Meta 区域

        # C-2. 扫描后续的数据包
        packet_len = 8 + BLOCK_SIZE  # Header(8) + Payload(32)

        while cursor + packet_len <= len(raw_data):
            chunk = raw_data[cursor: cursor + packet_len]
            # 尝试解析包 (lt_min 内部有 CRC 校验)
            try:
                pkt = lt_min.deserialize_lt_packet(chunk, BLOCK_SIZE)
                # 如果能解析出来，尝试加入解码器
                # 注意：deserialize 只是检查格式，decoder.add_packet 还会再次校验 CRC
                decoder.add_packet(pkt)
                img_pkt_count += 1
            except:
                pass  # 格式不对，说明这里的数据被严重破坏或不是包头

            cursor += packet_len

        print(f"  [图 #{idx + 1}] 提交数据包: {img_pkt_count} 个")
        total_valid_packets += img_pkt_count

    # ------------------------------------------------
    # 4. 最终结果
    # ------------------------------------------------
    print("\n" + "=" * 60)
    print("步骤4: 最终解码结果")
    print("=" * 60)

    stats = decoder.get_stats()
    print(f"[统计] 接收总包数(含重复): {stats['packets_received']}")
    print(f"[统计] CRC校验失败: {stats['packets_crc_failed']}")
    print(f"[统计] 有效块进度: {stats['progress']}")

    if decoder.is_decoded():
        recovered_bytes = decoder.reconstruct(verify_crc=True)
        print("\n🎉 成功! 文件完美复原！")
        # 注意：这里我们去掉切片 [:40]，直接打印全部内容，避免切断汉字导致报错
        try:
            print(f"原始内容: {secret_data.decode('utf-8')}")
            print(f"恢复内容: {recovered_bytes.decode('utf-8')}")
        except:
            # 万一数据坏了导致解不出来，就还是打印原始字节
            print(f"原始内容(raw): {secret_data}")
            print(f"恢复内容(raw): {recovered_bytes}")
        if recovered_bytes == secret_data:
            print(">>> 哈希校验一致 <<<")
    else:
        print("\n💀 失败! 数据不足或损坏严重。")

        # === 修复: 防止除以零 ===
        progress_str = stats['progress']  # 格式 "已解码/总数"
        try:
            # 安全解析
            decoded_count, total_count = map(int, progress_str.split('/'))
        except:
            decoded_count, total_count = 0, 0

        if total_count > 0:
            completion = (decoded_count / total_count) * 100
            print(f"当前恢复率: {completion:.1f}%")
        else:
            print("🔴 致命错误: 未能从任何图片中解析出 Meta 包(头部信息)。")
            print("   原因: JPEG压缩导致所有图片的文件头区域(前24字节)都发生了比特错误。")
            print("   建议:")
            print("   1. 提高 JPEG_QUALITY (例如 75)")
            print("   2. 增大 QIM_STEP (例如 100)")
            print("   3. 增加 SELECT_IMAGES 数量")
        # ========================