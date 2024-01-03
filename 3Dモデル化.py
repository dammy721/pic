# 深度推定,点群の生成,メッシュの作成

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

class DepthEstimator:
    def __init__(self, model_path):
        self.model = tf.saved_model.load(model_path)

    def estimate_depth(self, image_path):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = tf.image.resize(image_rgb, (256, 256))
        input_image = input_image[tf.newaxis, ...]
        depth_map = self.model(input_image)
        return depth_map[0].numpy()

def create_point_cloud(depth_map, image):
    points = []
    colors = []
    for v in range(image.shape[0]):
        for u in range(image.shape[1]):
            color = image[v, u]
            z = depth_map[v, u]
            if z > 0:
                points.append([u, v, z])
                colors.append(color)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors) / 255.0)
    return point_cloud

def create_mesh(point_cloud):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    return mesh

# 使用例
estimator = DepthEstimator('path_to_midas_model')
depth_map = estimator.estimate_depth('path_to_image.jpg')
image = cv2.imread('path_to_image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

point_cloud = create_point_cloud(depth_map, image_rgb)
mesh = create_mesh(point_cloud)

# 3Dモデルの表示
o3d.visualization.draw_geometries([mesh])
