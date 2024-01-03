import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

class SkeletonEstimator:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def estimate_skeleton(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (192, 192))
        input_data = np.expand_dims(image, axis=0)
        input_data = input_data.astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        keypoints = self.interpreter.get_tensor(self.output_details[0]['index'])

        return keypoints

# 使用例
estimator = SkeletonEstimator('path_to_movenet_model.tflite')
keypoints = estimator.estimate_skeleton('path_to_your_image.jpg')

# 結果の表示
for keypoint in keypoints[0][0]:
    y, x, score = keypoint
    if score > 0.4:
        cv2.circle(image, (int(x * image_width), int(y * image_height)), 5, (0, 255, 0), -1)

cv2.imshow("Skeleton Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
