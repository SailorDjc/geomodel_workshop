from PIL import Image
import imageio.v2 as iio
import os


# utils.plot_utils.control_visibility_with_layer_label接口会提供一个展示模型的界面窗口，窗口中有一个截图按钮，
# 本实例是将连续截图生成的图片，制成gif动画
def create_gif(image_list, gif_path, duration=1):
    """
    生成gif文件，原始图片仅支持png格式
    gif_name： 字符串
    duration:  gif图像时间间隔
    """
    frames = []
    for image_name in image_list:
        frames.append(iio.imread(image_name))
    iio.mimsave(gif_path, frames, format='GIF', duration=duration)


if __name__ == '__main__':
    image_list = [r'E:\Pycode\11-22-GeoSci\geomodel_workshop-main\output\pic_1704188291.png'
        , r'E:\Pycode\11-22-GeoSci\geomodel_workshop-main\output\pic_1704188334.png'
        , r'E:\Pycode\11-22-GeoSci\geomodel_workshop-main\output\pic_1704188338.png'
        , r'E:\Pycode\11-22-GeoSci\geomodel_workshop-main\output\pic_1704188341.png']
    create_gif(image_list, r'E:\Pycode\11-22-GeoSci\geomodel_workshop-main\output\animation.gif')
