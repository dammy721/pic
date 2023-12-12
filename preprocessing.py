import cv2

# from PIL import Image
# import numpy as np
# import os

# def create_random_jpg(filename, width, height):
#     # ランダムなピクセル値を持つ配列を生成
#     array = np.random.rand(height, width, 3) * 255
#     array = array.astype(np.uint8)

#     # 配列から画像を生成
#     image = Image.fromarray(array, 'RGB')

#     # JPEG形式で保存
#     image.save(filename)

# from PIL import Image
# import matplotlib.pyplot as plt

# # 画像を生成
# create_random_jpg('random_image.jpg', 300, 300)

# # PILを使って画像を読み込む
# image = Image.open('random_image.jpg')

# # Matplotlibを使って画像を表示
# plt.imshow(image)  # カラー画像の場合は cmap='gray' は不要です。
# plt.axis('off')  # 軸を非表示にする
# plt.show()


class ImagePreprocessor:
    def __init__(self, image_path):
        # 指定されたパスから画像を読み込む
        self.image = cv2.imread(image_path)

    def convert_to_grayscale(self):
        # 画像をグレースケールに変換
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.image

    def apply_gaussian_blur(self, kernel_size=(5, 5)):
        # 画像にガウシアンブラー（平滑化フィルタ）を適用
        self.image = cv2.GaussianBlur(self.image, kernel_size, 0)
        return self.image

    def detect_edges(self, threshold1, threshold2):
        # Cannyエッジ検出器を使用して画像のエッジを検出
        self.image = cv2.Canny(self.image, threshold1, threshold2)
        return self.image

    def apply_threshold(self, threshold, max_value=255):
        # 二値化を適用（画像を白黒に）
        _, self.image = cv2.threshold(self.image, threshold, max_value, cv2.THRESH_BINARY)
        return self.image

    def resize_image(self, width, height):
        # 画像のサイズを変更
        self.image = cv2.resize(self.image, (width, height))
        return self.image

    def equalize_histogram(self):
        # 画像のヒストグラムを均一化し、コントラストを改善
        self.image = cv2.equalizeHist(self.image)
        return self.image

    def get_image(self):
        # 現在の画像を返す
        return self.image

import matplotlib.pyplot as plt

image_paths = ['random_image.jpg'] 
# 画像の前処理
for path in image_paths:
    preprocessor = ImagePreprocessor(path)
    preprocessor.convert_to_grayscale()
    preprocessor.apply_gaussian_blur()
    preprocessor.detect_edges(100, 200)
    preprocessor.apply_threshold(127)
    preprocessor.resize_image(128, 128)
    preprocessor.equalize_histogram()
    processed_image = preprocessor.get_image()

    # 処理された画像を表示
    plt.imshow(processed_image, cmap='gray')
    plt.axis('off')
    plt.show()

