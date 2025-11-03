import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import av
import argparse
from PIL import Image
import numpy as np


def rebuild_video_with_i_frames(
        original_video: str,
        i_frames_dir: str,
        output_video: str,
        codec: str = "libx264",
        crf: int = 18
):
    """
    用修改后的 I 帧替换原视频中的 I 帧
    """
    # 读取所有修改后的 I 帧(按 pts 排序)
    i_frame_files = {}
    for fname in os.listdir(i_frames_dir):
        if fname.startswith("I_pts_") and fname.endswith(".jpg"):
            pts = int(fname.replace("I_pts_", "").replace(".jpg", ""))
            i_frame_files[pts] = os.path.join(i_frames_dir, fname)

    print(f"找到 {len(i_frame_files)} 个修改后的 I 帧")

    # 打开原视频
    input_container = av.open(original_video)
    input_stream = input_container.streams.video[0]

    # 创建输出视频
    output_container = av.open(output_video, 'w')
    output_stream = output_container.add_stream(codec, rate=input_stream.average_rate)
    output_stream.width = input_stream.width
    output_stream.height = input_stream.height
    output_stream.pix_fmt = 'yuv420p'
    output_stream.options = {'crf': str(crf)}

    frame_count = 0
    replaced_count = 0

    for frame in input_container.decode(input_stream):
        pts = frame.pts if frame.pts is not None else 0

        # 判断是否是 I 帧且有对应的修改图片
        if pts in i_frame_files:
            # 读取修改后的 I 帧
            img = Image.open(i_frame_files[pts])
            img_array = np.array(img)

            # 转换为 VideoFrame
            new_frame = av.VideoFrame.from_ndarray(img_array, format='rgb24')
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

            # 编码修改后的帧
            for packet in output_stream.encode(new_frame):
                output_container.mux(packet)

            replaced_count += 1
            print(f"替换 I 帧: pts={pts}")
        else:
            # 保持原帧
            frame.pts = frame.pts
            for packet in output_stream.encode(frame):
                output_container.mux(packet)

        frame_count += 1

    # 刷新编码器
    for packet in output_stream.encode():
        output_container.mux(packet)

    input_container.close()
    output_container.close()

    print(f"完成! 总帧数: {frame_count}, 替换 I 帧: {replaced_count}")
    print(f"输出视频: {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用修改后的 I 帧重建视频")
    parser.add_argument("--original", required=True, help="原始视频路径")
    parser.add_argument("--i-frames", required=True, help="修改后的 I 帧文件夹路径")
    parser.add_argument("--output", required=True, help="输出视频路径")
    parser.add_argument("--codec", default="libx264", help="视频编码器(默认 libx264)")
    parser.add_argument("--crf", type=int, default=18, help="CRF 质量参数(默认18,越小质量越高)")

    args = parser.parse_args()
    rebuild_video_with_i_frames(args.original, args.i_frames, args.output, args.codec, args.crf)