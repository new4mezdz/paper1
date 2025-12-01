# -*- coding: utf-8 -*-
import cv2
import numpy as np
import zlib
import struct


# ==========================================
# QIM 核心算法
# ==========================================
def qim_embed_scalar(val, bit, step):
    quant_idx = round(val / step)
    if bit == 0:
        if quant_idx % 2 != 0:
            if (val / step) >= quant_idx:
                quant_idx += 1
            else:
                quant_idx -= 1
    else:
        if quant_idx % 2 == 0:
            if (val / step) >= quant_idx:
                quant_idx += 1
            else:
                quant_idx -= 1
    return quant_idx * step


def qim_extract_scalar(val, step):
    quant_idx = round(val / step)
    return quant_idx % 2


# ==========================================
# 喷泉包结构
# ==========================================
def create_fountain_packet(packet_id, data_str):
    """
    创建喷泉包：
    格式: [包ID(4字节)] + [数据长度(2字节)] + [数据] + [CRC32(4字节)]
    """
    data_bytes = data_str.encode('utf-8')
    data_len = len(data_bytes)
    crc = zlib.crc32(data_bytes)

    # 打包：ID(4) + 长度(2) + 数据 + CRC(4)
    packet = struct.pack('>I', packet_id)  # 包ID
    packet += struct.pack('>H', data_len)  # 数据长度
    packet += data_bytes  # 数据内容
    packet += struct.pack('>I', crc)  # CRC校验

    return packet, crc


def parse_fountain_packet(packet_bytes):
    """
    解析喷泉包
    返回: (packet_id, data, crc, is_valid)
    """
    if len(packet_bytes) < 10:  # 最小包长度
        return None, None, None, False

    try:
        # 解包头部
        packet_id = struct.unpack('>I', packet_bytes[0:4])[0]
        data_len = struct.unpack('>H', packet_bytes[4:6])[0]

        # 检查长度
        if len(packet_bytes) < 10 + data_len:
            return packet_id, None, None, False

        # 提取数据和CRC
        data = packet_bytes[6:6 + data_len]
        stored_crc = struct.unpack('>I', packet_bytes[6 + data_len:10 + data_len])[0]

        # 计算实际CRC
        calculated_crc = zlib.crc32(data)
        is_valid = (stored_crc == calculated_crc)

        return packet_id, data.decode('utf-8'), stored_crc, is_valid
    except:
        return None, None, None, False


def bytes_to_bits(data):
    """字节转比特数组"""
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return np.array(bits, dtype=np.uint8)


def bits_to_bytes(bits):
    """比特数组转字节"""
    # 补齐到8的倍数
    remainder = len(bits) % 8
    if remainder != 0:
        bits = np.concatenate([bits, np.zeros(8 - remainder, dtype=np.uint8)])

    bytes_data = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        bytes_data.append(byte)
    return bytes(bytes_data)


# ==========================================
# 图像嵌入与提取（支持多个包）
# ==========================================
def embed_multiple_packets(img_gray, packets_list, step=50):
    """
    重复嵌入多个包
    packets_list: 包字节数组的列表
    """
    h, w = img_gray.shape
    img_float = img_gray.astype(float)
    stego_img = img_float.copy()

    # 将所有包连接起来
    all_data = b''.join(packets_list)
    all_bits = bytes_to_bits(all_data)

    # 计算容量
    capacity = (h // 8) * (w // 8)

    print(f"[嵌入] 图像容量: {capacity} bits")
    print(f"[嵌入] 总数据: {len(all_bits)} bits ({len(all_data)} bytes)")
    print(f"[嵌入] 包数量: {len(packets_list)}")

    if len(all_bits) > capacity:
        print(f"[警告] 数据超过容量! 将被截断")
        all_bits = all_bits[:capacity]

    bit_idx = 0
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            if bit_idx >= len(all_bits):
                break

            block = img_float[y:y + 8, x:x + 8]
            dct_block = cv2.dct(block)

            original_coeff = dct_block[4, 3]
            bit = all_bits[bit_idx]
            new_coeff = qim_embed_scalar(original_coeff, bit, step)
            dct_block[4, 3] = new_coeff

            stego_block = cv2.idct(dct_block)
            stego_img[y:y + 8, x:x + 8] = stego_block

            bit_idx += 1

    stego_img_uint8 = np.clip(stego_img, 0, 255).astype(np.uint8)
    return stego_img_uint8, len(all_bits)


def extract_multiple_packets(img_gray, num_bits, packet_size, step=50):
    """
    提取多个包
    packet_size: 单个包的字节大小
    """
    h, w = img_gray.shape
    img_float = img_gray.astype(float)

    extracted_bits = []

    for y in range(0, h, 8):
        for x in range(0, w, 8):
            if len(extracted_bits) >= num_bits:
                break

            block = img_float[y:y + 8, x:x + 8]
            dct_block = cv2.dct(block)
            coeff = dct_block[4, 3]
            bit = qim_extract_scalar(coeff, step)
            extracted_bits.append(bit)

    extracted_bits = np.array(extracted_bits[:num_bits], dtype=np.uint8)
    extracted_bytes = bits_to_bytes(extracted_bits)

    # 按包大小分割
    packets = []
    for i in range(0, len(extracted_bytes), packet_size):
        packet_data = extracted_bytes[i:i + packet_size]
        if len(packet_data) == packet_size:
            packets.append(packet_data)

    return packets


# ==========================================
# 主测试程序
# ==========================================
if __name__ == "__main__":
    # 1. 读取图片
    print("=" * 60)
    print("步骤1: 读取图片")
    print("=" * 60)

    img_path = r"D:\paper data\stego_images\I_pts_364.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"[错误] 无法读取图片: {img_path}")
        print("[提示] 将使用随机图片代替测试")
        img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    else:
        print(f"[成功] 图片尺寸: {img.shape}")

    # 裁剪到8的倍数
    h, w = img.shape
    h = h // 8 * 8
    w = w // 8 * 8
    img = img[:h, :w]
    print(f"[调整] 裁剪后尺寸: {img.shape}")

    # 2. 创建一个喷泉包
    print("\n" + "=" * 60)
    print("步骤2: 创建喷泉包")
    print("=" * 60)

    test_data = "这是一个测试喷泉包的数据内容"
    packet_bytes, original_crc = create_fountain_packet(packet_id=1, data_str=test_data)
    packet_size = len(packet_bytes)

    print(f"[包信息] 包ID: 1")
    print(f"[包信息] 数据: {test_data}")
    print(f"[包信息] 包大小: {packet_size} bytes ({packet_size * 8} bits)")
    print(f"[包信息] CRC32: {original_crc}")

    # 3. 重复嵌入N次
    print("\n" + "=" * 60)
    print("步骤3: 重复嵌入")
    print("=" * 60)

    repeat_count = 10  # 重复10次
    packets_to_embed = [packet_bytes] * repeat_count

    print(f"[配置] 重复次数: {repeat_count}")
    print(f"[配置] 总数据量: {packet_size * repeat_count} bytes")
    print(f"[配置] QIM步长: 50")

    stego_img, embedded_bits = embed_multiple_packets(img, packets_to_embed, step=50)

    # 计算PSNR
    mse = np.mean((img.astype(float) - stego_img.astype(float)) ** 2)
    if mse > 0:
        psnr = 10 * np.log10(255 ** 2 / mse)
        print(f"[质量] PSNR: {psnr:.2f} dB")

    # 3.5. 模拟JPEG压缩攻击
    print("\n" + "=" * 60)
    print("步骤3.5: JPEG压缩攻击")
    print("=" * 60)

    quality = 50 # 质量参数，可以调整
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', stego_img, encode_param)
    attacked_img = cv2.imdecode(encimg, 0)

    print(f"[压缩] JPEG质量: {quality}")

    # 4. 提取包（改成从attacked_img提取）
    print("\n" + "=" * 60)
    print("步骤4: 提取包")
    print("=" * 60)

    extracted_packets = extract_multiple_packets(attacked_img, embedded_bits, packet_size, step=50)

    print(f"[提取] 期望包数: {repeat_count}")
    print(f"[提取] 实际提取: {len(extracted_packets)} 个")

    # 5. 验证每个包
    print("\n" + "=" * 60)
    print("步骤5: 验证包完整性")
    print("=" * 60)

    # 在步骤5验证时，加上更严格的检查

    valid_count = 0
    correct_count = 0  # 新增：内容正确的包

    for i, packet_data in enumerate(extracted_packets):
        pid, data, crc, is_valid = parse_fountain_packet(packet_data)

        # 检查是否是我们嵌入的包
        is_correct = (pid == 1 and data == test_data and crc == original_crc)

        status = "✓" if is_valid else "✗"
        print(f"\n包 #{i + 1}:")
        print(f"  CRC校验: {status}")
        print(f"  内容正确: {'✓' if is_correct else '✗'}")
        print(f"  包ID: {pid} (期望: 1)")
        if data is not None:
            print(f"  数据: {data[:20]}... (期望: {test_data[:20]}...)")
        else:
            print(f"  数据: [解析失败] (期望: {test_data[:20]}...)")
        print(f"  CRC: {crc} (期望: {original_crc})")

        if is_valid:
            valid_count += 1
        if is_correct:
            correct_count += 1

    # 统计
    print(f"\nCRC通过: {valid_count} 个")
    print(f"内容正确: {correct_count} 个")
    print(f"真实成功率: {correct_count / repeat_count * 100:.1f}%")
    if valid_count == repeat_count:
        print("\n🎉 完美！所有包都完整提取！")
    elif valid_count > repeat_count * 0.8:
        print("\n✅ 很好！大部分包完整，配合喷泉码足够恢复！")
    elif valid_count > 0:
        print("\n⚠️  部分包损坏，可能需要更多冗余")
    else:
        print("\n❌ 全部损坏，需要调整参数")