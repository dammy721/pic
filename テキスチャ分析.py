import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops

def analyze_texture(image_path):
    # 画像を読み込み、グレースケールに変換
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # GLCMの計算
    glcm = greycomatrix(image, distances=[1], angles=[0], symmetric=True, normed=True)

    # GLCMの統計的特徴を取得
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]

    return contrast, dissimilarity, homogeneity, energy, correlation

# 使用例
features = analyze_texture('path_to_texture_image.jpg')
print("Contrast:", features[0])
print("Dissimilarity:", features[1])
print("Homogeneity:", features[2])
print("Energy:", features[3])
print("Correlation:", features[4])
