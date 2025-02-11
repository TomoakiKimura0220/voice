#!/usr/bin/env python3
import json
from fer import FER

class ExpressionRecognizer:
    def __init__(self, mtcnn=True):
        """
        ExpressionRecognizer のコンストラクタ

        Parameters:
            mtcnn (bool): True にすると MTCNN を用いた顔検出を利用（認識精度向上が期待できる）
        """
        self.detector = FER(mtcnn=mtcnn)

    def recognize(self, image):
        """
        入力画像に対して表情認識を実施し、画像中心に最も近い顔の認識結果をJSON形式で返す

        Parameters:
            image (numpy.ndarray): OpenCV形式の画像（BGR）

        Returns:
            str: JSON形式の認識結果例
                {
                    "face": {
                        "box": [x, y, w, h],
                        "dominant_emotion": "happy",
                        "score": 0.92,
                        "all_emotions": {
                            "angry": 0.01,
                            "disgust": 0.00,
                            "fear": 0.02,
                            "happy": 0.92,
                            "sad": 0.01,
                            "surprise": 0.03,
                            "neutral": 0.01
                        }
                    }
                }
                ※ 顔が検出されなかった場合は {"face": null} を返す
        """
        results = self.detector.detect_emotions(image)
        if not results:
            return json.dumps({"face": None}, ensure_ascii=False, indent=2)

        # 画像サイズから画像の中心座標を求める
        height, width = image.shape[:2]
        image_center = (width / 2, height / 2)

        # 各顔の中心と画像中心との距離の2乗を計算する関数
        def center_distance(face):
            box = face.get("box", [0, 0, 0, 0])
            x, y, w, h = box
            face_center = (x + w / 2, y + h / 2)
            dx = face_center[0] - image_center[0]
            dy = face_center[1] - image_center[1]
            return dx * dx + dy * dy  # 平方根を取らなくても比較可能

        # 画像中心に最も近い顔を選択
        best_face = min(results, key=center_distance)

        box = best_face.get("box")
        emotions = best_face.get("emotions", {})
        if emotions:
            dominant_emotion = max(emotions, key=emotions.get)
            score = emotions[dominant_emotion]
        else:
            dominant_emotion = "unknown"
            score = 0.0

        face_data = {
            "box": box,
            "dominant_emotion": dominant_emotion,
            "score": score,
            "all_emotions": emotions
        }
        output = {"face": face_data}
        return json.dumps(output, ensure_ascii=False, indent=2)

# テスト用メイン（モジュールとして利用する場合は不要）
if __name__ == "__main__":
    import cv2

    # カメラから1フレーム取得してテストする例
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        recognizer = ExpressionRecognizer(mtcnn=True)
        json_result = recognizer.recognize(frame)
        print("認識結果:")
        print(json_result)
    else:
        print("画像の取得に失敗しました。")
