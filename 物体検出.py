import cv2
import numpy as np

class YOLODetector:
    def __init__(self, config_path, weights_path, class_names_path):
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.classes = open(class_names_path).read().strip().split('\n')

    def detect_objects(self, image_path):
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # YOLOに画像を入力
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)

        # 検出結果の処理
        boxes = []
        confidences = []
        class_ids = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # バウンディングボックスの座標を取得
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 非最大抑制
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return image

# 使用例
detector = YOLODetector('path_to_cfg_file.cfg', 'path_to_weights.weights', 'path_to_classes.names')
detected_image = detector.detect_objects('path_to_your_image.jpg')
cv2.imshow('Detected Objects', detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
