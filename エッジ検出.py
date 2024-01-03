###エッジ検出Canny,Sobel,Laplacian
import cv2
import matplotlib.pyplot as plt

class EdgeDetection:
    def __init__(self, image_path):
        # グレースケールで画像を読み込む
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    def canny_edge_detection(self, threshold1=100, threshold2=200):
        # Cannyエッジ検出を適用
        return cv2.Canny(self.image, threshold1, threshold2)

    def sobel_edge_detection(self, ksize=5):
        # Sobelエッジ検出を適用（x方向とy方向）
        sobelx = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=ksize)
        return sobelx, sobely

    def laplacian_edge_detection(self):
        # Laplacianエッジ検出を適用
        return cv2.Laplacian(self.image, cv2.CV_64F)

# 使用例
edge_detector = EdgeDetection('random_image.jpg')
canny_edges = edge_detector.canny_edge_detection()
sobelx, sobely = edge_detector.sobel_edge_detection()
laplacian_edges = edge_detector.laplacian_edge_detection()

# 結果の表示
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.imshow(canny_edges, cmap='gray')
plt.title('Canny'), plt.axis('off')
plt.subplot(1, 3, 2), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.axis('off')
plt.subplot(1, 3, 3), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.axis('off')
plt.show()

plt.imshow(laplacian_edges, cmap='gray')
plt.title('Laplacian'), plt.axis('off')
plt.show()
