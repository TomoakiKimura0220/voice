# テスト用プログラムファイル

#!/usr/bin/env python3
import cv2
from fer import FER
import time

def main():
    # FER の顔表情認識器を初期化（mtcnn=True で MTCNN による顔検出を利用可能）
    detector = FER(mtcnn=True)
    
    # デフォルトカメラ（0）をキャプチャする
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("カメラが開けませんでした。")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームの取得に失敗しました。")
            break

        # 表情認識を実施
        results = detector.detect_emotions(frame)


        # 表情認識を実施
        results = detector.detect_emotions(frame)
        
        # 検出された各顔に対して
        for result in results:
            (x, y, w, h) = result["box"]
            emotions = result["emotions"]
            # 感情スコアが最大のものを抽出
            if emotions:
                emotion, score = max(emotions.items(), key=lambda item: item[1])
            else:
                emotion, score = "unknown", 0.0

            # 顔領域に四角形を描画
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 認識結果を描画
            text = f"{emotion} ({score:.2f})"
            cv2.putText(
                            frame, 
                            text, 
                            (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, 
                            (0, 255, 0),
                            2
                        )

        # フレームを表示
        cv2.imshow("リアルタイム表情認識", frame)

        # 結果があれば、各顔ごとに主な感情とスコアをコンソール出力
        if results:
            for idx, face in enumerate(results, start=1):
                emotions = face["emotions"]
                dominant_emotion = max(emotions, key=emotions.get)
                score = emotions[dominant_emotion]
                print(f"Face {idx}: {dominant_emotion} ({score:.2f})")
        else:
            print("顔が検出されませんでした。")
        
        # 連続出力だとコンソールが大量に埋まるため、適宜待機（例：0.5秒）
        time.sleep(0.5)
        
        # キー 'q' が押されたらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()