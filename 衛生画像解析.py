### 土地被覆分類、植生指数NDVI計算、建物検出、変化検出、水域検出、温度分析
import rasterio
import numpy as np
from sklearn.cluster import KMeans

class SatelliteImageAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        with rasterio.open(image_path) as src:
            self.image = src.read()
            self.profile = src.profile

    def calculate_ndvi(self):
        # NDVI計算（バンド4: NIR, バンド3: Red）
        nir = self.image[3].astype(float)
        red = self.image[2].astype(float)
        ndvi = (nir - red) / (nir + red)

        return ndvi

    def land_cover_classification(self, n_clusters=5):
        # 土地被覆分類
        reshaped_img = self.image.reshape(-1, self.image.shape[0])
        kmeans = KMeans(n_clusters=n_clusters).fit(reshaped_img)
        clustered = kmeans.labels_.reshape(self.image.shape[1], self.image.shape[2])

        return clustered

    def detect_buildings(self):
        # 建物検出（単純化のための例）
        # 通常はより複雑なアルゴリズムが必要
        red = self.image[2]
        building_mask = red > np.percentile(red, 95)

        return building_mask

    def change_detection(self, image_path2):
        # 画像2の読み込み
        with rasterio.open(image_path2) as src:
            image2 = src.read()

        # 変化検出（単純な差分計算）
        change = np.abs(self.image - image2)

        return change

    def water_detection(self):
        # 水域検出（NIRと赤色バンドの比較）
        nir = self.image[3].astype(float)
        red = self.image[2].astype(float)
        ndwi = (nir - red) / (nir + red)  # Normalized Difference Water Index

        water_mask = ndwi > 0.3  # ある閾値以上を水域とする

        return water_mask

    def temperature_analysis(self):
        # 温度分析（熱赤外線バンドの解析）
        # ここでは仮の方法を示す
        thermal_band = self.image[10].astype(float)  # 例としてバンド10を使用
        temperature = thermal_band * 0.1  # 仮の変換式

        return temperature

# 使用例
analyzer = SatelliteImageAnalyzer('path_to_satellite_image1.tif')
change = analyzer.change_detection('path_to_satellite_image2.tif')
water = analyzer.water_detection()
temperature = analyzer.temperature_analysis()

# 結果の保存や表示は、追加でコーディングが必要
