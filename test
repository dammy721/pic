import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

class SuperResolution:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def enhance_image(self, image_path):
        # 画像の読み込みと前処理
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_details[0]['shape'][2], self.input_details[0]['shape'][1]))
        input_data = np.expand_dims(image, axis=0).astype(np.float32)

        # 超解像度モデルによる画像の強化
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output_data[0]

# 使用例
sr = SuperResolution('path_to_sr_model.tflite')
high_res_image = sr.enhance_image('path_to_low_res_image.jpg')

# 結果の表示
cv2.imshow('High Resolution Image', cv2.cvtColor(high_res_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
