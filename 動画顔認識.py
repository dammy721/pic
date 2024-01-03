import cv2

def recognize_faces_in_video(video_path):
    # 顔検出用のカスケード分類器をロード
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 動画キャプチャの初期化
    cap = cv2.VideoCapture(video_path)

    while True:
        # 動画からフレームを読み込む
        ret, frame = cap.read()
        if not ret:
            break

        # グレースケール画像に変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 顔の検出
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # 検出された顔に四角を描画
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 結果の表示
        cv2.imshow('Face Recognition', frame)

        # 'q'キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 使用例
recognize_faces_in_video('path_to_video.mp4')
