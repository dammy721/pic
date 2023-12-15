import cv2
from google.colab.patches import cv2_imshow  # Colab環境での画像表示用

def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return image

# 画像ファイルのパス
image_path = '/content/sample_data/human.png'  # 画像パスを修正

# 顔検出を実行
detected_image = detect_faces(image_path)

# 結果を表示
cv2_imshow(detected_image)  # Colab用の表示関数を使用
