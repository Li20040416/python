import rasterio
import numpy as np
import os
from PIL import Image

# 避免 Qt 后端错误
import matplotlib
matplotlib.use('TkAgg')  # 或改为 'Agg'（无 GUI 环境）
import matplotlib.pyplot as plt


def process_sentinel2_image(input_path, output_dir='output', output_format='jpg', display=True, brightness_factor=1.2):
    """
    处理 Sentinel-2 多波段影像数据，生成 RGB 图像，支持亮度调节

    参数:
        input_path (str): 输入的多波段 TIFF 路径
        output_dir (str): 输出目录
        output_format (str): 输出格式（'jpg' 或 'tif'）
        display (bool): 是否显示图像
        brightness_factor (float): 亮度增强因子（默认 1.2）

    返回:
        rgb_image (numpy.ndarray): RGB 图像数组
        output_path (str): 输出文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 读取多波段影像
        with rasterio.open(input_path) as src:
            bands = src.read()
            profile = src.profile

        if bands.shape[0] < 4:
            raise ValueError("输入文件需至少包含 4 个波段（B2, B3, B4, B8）")

        def process_band(band_data):
            return np.clip(band_data.astype(float) / 10000 * 255, 0, 255).astype(np.uint8)

        # 假设波段顺序为 B2, B3, B4, B8...
        blue = process_band(bands[0])   # B2
        green = process_band(bands[1])  # B3
        red = process_band(bands[2])    # B4

        # 合并 RGB 图像
        rgb_image = np.dstack((red, green, blue))

        # ⭐️ 提升亮度
        rgb_image = np.clip(rgb_image.astype(float) * brightness_factor, 0, 255).astype(np.uint8)

        # 准备输出文件名
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        if output_format.lower() == 'tif':
            output_path = os.path.join(output_dir, f"{base_name}_RGB.tif")
            profile.update(count=3, dtype='uint8')
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(rgb_image[:, :, 0], 1)
                dst.write(rgb_image[:, :, 1], 2)
                dst.write(rgb_image[:, :, 2], 3)
        else:
            output_path = os.path.join(output_dir, f"{base_name}_RGB.jpg")
            Image.fromarray(rgb_image).save(output_path, quality=95)

        # 可视化
        if display:
            plt.figure(figsize=(12, 8))
            plt.imshow(rgb_image)
            plt.title(f'Processed Sentinel-2 Image\n{base_name} (Brightness × {brightness_factor})')
            plt.axis('off')
            plt.show()

        print(f"✅ 处理成功！图像保存至: {output_path}")
        return rgb_image, output_path

    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        return None, None


if __name__ == "__main__":
    # 输入文件路径（用原始字符串避免转义问题）
    input_file = r"D:\存储\QQ\250834326\FileRecv\2019_1101_nofire_B2348_B12_10m_roi.tif"

    if not os.path.exists(input_file):
        print(f"❌ 错误：文件不存在 - {input_file}")
    else:
        # 调用处理函数，提升亮度，保存并显示图像
        rgb_result, out_path = process_sentinel2_image(
            input_path=input_file,
            output_dir='processed_output',
            output_format='jpg',     # 可选：'jpg' 或 'tif'
            display=True,            # 设置 False 可跳过显示
            brightness_factor=1.3    # ⭐️ 可调节：1.0 原始亮度，1.2 略亮，1.5 明显亮
        )
