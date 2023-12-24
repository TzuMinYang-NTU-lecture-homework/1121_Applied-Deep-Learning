import json
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters import threshold_local
import numpy as np

#with open("train.json", "r") as f:
#    data = json.loads(f.read())
#tags = [tag for entity in data for tag in entity["input"]]
## print(tags)
#with open("tags.txt", "w") as f:
#    f.write(" ".join(tags))
    
# 打開圖片
#img = Image.open("book_black.jpg")
#smoothed_image = gaussian_filter(img, sigma=2)

# 將圖片轉換為灰度
#gray_img = img.convert("L")
#
## 將灰度圖片轉換為黑白圖片（二值化）
#bw_img = gray_img.point(lambda x: 0 if x < 200 else 255, '1')
#
## 保存黑白圖片
#bw_img.save("book_black2.jpg")


def convert_png_to_jpg(input_path, output_path):
    # 打开PNG图像
    png_image = Image.open(input_path)

    # 创建白色底的新图像
    jpg_image = Image.new("RGB", png_image.size, (255, 255, 255))

    # 判断是否存在透明度通道
    if len(png_image.split()) == 4:
        # 将PNG图像粘贴到新图像上（透明部分变为白色）
        jpg_image.paste(png_image, mask=png_image.split()[3])
    else:
        jpg_image.paste(png_image)

    # 保存为JPEG格式
    jpg_image.save(output_path, "JPEG")

# 用法示例
input_path = "book.png"
output_path = "output.jpg"
convert_png_to_jpg(input_path, output_path)