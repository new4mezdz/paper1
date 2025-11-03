def calculate_text_size():
    """
    提示用户输入文本，并计算其大小（字节和比特）。
    """
    # 1. 提示用户输入自定义的文本信息
    # input() 函数会读取用户在命令行中输入的一行内容
    user_text = input("请输入您想要嵌入的水印信息: ")

    # 2. 将文本字符串编码为字节 (bytes)
    # 我们使用 'utf-8'，这是最通用和标准的文本编码格式。
    # .encode() 方法会将字符串转换为字节序列。
    text_in_bytes = user_text.encode('utf-8')

    # 3. 计算大小
    # len() 函数作用于字节序列时，会返回字节的数量。
    size_in_bytes = len(text_in_bytes)

    # 1 字节 = 8 比特
    size_in_bits = size_in_bytes * 8

    # 4. 打印出清晰的结果
    print("\n" + "=" * 20 + " 分析结果 " + "=" * 20)
    print(f"您输入的文本是: '{user_text}'")
    print(f"文本的长度 (字符数): {len(user_text)}")
    print("-" * 52)
    print(f"编码后的字节大小: {size_in_bytes} 字节 (Bytes)")
    print(f"换算后的比特大小: {size_in_bits} 比特 (Bits)")
    print("=" * 52)
    print("\n说明: 这个 '字节大小' 就是您的 LT 编码器将要处理的源文件大小 F。")


# --- 主程序入口 ---
if __name__ == '__main__':
    calculate_text_size()