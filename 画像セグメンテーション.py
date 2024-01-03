import cv2
import numpy as np

def segment_image(image_path):
    # 画像を読み込み
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 特定の色範囲を定義（例：青色）
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # 色範囲に基づいてマスクを作成
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # マスクを適用して結果を取得
    result = cv2.bitwise_and(image, image, mask=mask)

    # 結果を表示
    cv2.imshow('Original Image', image)
    cv2.imshow('Segmented Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用例
segment_image('path_to_image.jpg')
