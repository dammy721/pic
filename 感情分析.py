from fer import FER
import matplotlib.pyplot as plt

# 画像ファイルのパスを指定
test_image_one = plt.imread("画像ファイルパス")

# 感情検出器の初期化
emo_detector = FER(mtcnn=True)

# 画像から感情を検出
captured_emotions = emo_detector.detect_emotions(test_image_one)
print(captured_emotions)

# 画像内で最も支配的な感情とそのスコアを取得
dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)

# 画像を表示
plt.imshow(test_image_one)
plt.show()

# 支配的な感情とスコアを出力
print(dominant_emotion, emotion_score)
