import random
import zlib

def create_fountain_packet(packet_id, data):
    """创建一个喷泉包，包含ID和数据"""
    return {
        'id': packet_id,
        'data': data,
        'crc': zlib.crc32(data)
    }

def verify_crc(packet):
    """验证CRC校验值"""
    calculated_crc = zlib.crc32(packet['data'])
    return calculated_crc == packet['crc']

# 1. 生成几个喷泉包
print("=== 步骤1: 生成喷泉包 ===")
packets = []
for i in range(5):
    data = f"这是第{i}号数据包的内容".encode('utf-8')
    packet = create_fountain_packet(i, data)
    packets.append(packet)
    print(f"包{i}: CRC={packet['crc']}, 数据={packet['data'].decode('utf-8')}")

# 2. 验证所有包都是正常的
print("\n=== 步骤2: 验证原始包 ===")
for packet in packets:
    is_valid = verify_crc(packet)
    print(f"包{packet['id']}: {'✓ 校验通过' if is_valid else '✗ 校验失败'}")

# 3. 人为修改某些包
print("\n=== 步骤3: 人为制造错误 ===")
# 修改包1的数据（改变一个字节）
original_data_1 = packets[1]['data']
packets[1]['data'] = b"Modified data for packet 1"
print(f"包1: 将数据从 '{original_data_1.decode('utf-8')}' 改为 '{packets[1]['data'].decode('utf-8')}'")

# 修改包3的数据（只改变一个字符）
original_data_3 = packets[3]['data']
packets[3]['data'] = packets[3]['data'][:-1] + b'X'
print(f"包3: 将最后一个字符改为 'X'")

# 包2和包4保持不变

# 4. 再次验证所有包
print("\n=== 步骤4: 验证修改后的包 ===")
for packet in packets:
    is_valid = verify_crc(packet)
    status = '✓ 校验通过' if is_valid else '✗ 校验失败'
    print(f"包{packet['id']}: {status}")
    if not is_valid:
        print(f"  预期CRC: {packet['crc']}")
        print(f"  实际CRC: {zlib.crc32(packet['data'])}")

# 5. 测试更细微的错误
print("\n=== 步骤5: 测试单字节错误 ===")
test_data = b"Test single byte error"
test_packet = create_fountain_packet(99, test_data)
print(f"原始数据: {test_data}")
print(f"原始CRC: {test_packet['crc']}")

# 只改变一个位
modified_data = bytearray(test_data)
modified_data[5] = modified_data[5] ^ 0x01  # 翻转一个位
test_packet['data'] = bytes(modified_data)

print(f"修改后数据: {test_packet['data']}")
print(f"修改后CRC: {zlib.crc32(test_packet['data'])}")
print(f"CRC能否检测: {'✗ 检测到错误' if not verify_crc(test_packet) else '✓ 未检测到错误'}")