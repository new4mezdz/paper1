import cv2
import numpy as np


def create_and_cut_blue_rect():
    # 1. 定义尺寸 (高, 宽)
    h, w = 600, 800

    # 2. 创建一个全黑的画布 (Height, Width, 3通道)
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # 3. 填充纯蓝色
    # 注意：OpenCV 的颜色顺序是 BGR (Blue, Green, Red)
    # 所以 (255, 0, 0) 就是纯蓝
    img[:] = (255, 0, 0)

    # 保存一下原图，方便对比
    cv2.imwrite("blue_origin.png", img)
    print("✅ 原图已生成: blue_origin.png")

    # ==========================
    # 开始模拟“右上角缺失”
    # ==========================

    # 定义剪切比例 (比如切掉 40%)
    cut_ratio = 0.4

    # 计算三角形的三个顶点
    # 顶点1：顶部靠左一点
    p1 = (int(w * (1 - cut_ratio)), 0)
    # 顶点2：右侧靠下一点
    p2 = (w, int(h * cut_ratio))
    # 顶点3：右上角 (必须包含这个角)
    p3 = (w, 0)

    # 创建顶点数组
    triangle_cnt = np.array([p1, p2, p3], np.int32)

    # 把这个三角形区域涂成黑色 (0, 0, 0)
    # 黑色代表：这部分数据丢了，SIFT 提不到点，LT 码收不到包
    cv2.fillPoly(img, [triangle_cnt], (0, 0, 0))

    # 保存结果
    cv2.imwrite("blue_cut_top_right.png", img)
    print("✅ 裁剪图已生成: blue_cut_top_right.png")


if __name__ == "__main__":
    create_and_cut_blue_rect()