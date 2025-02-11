# 画像ファイルを入力データとして使用する場合

# import cv2
# # import time #処理時間の計測用
# from expression_recognition import ExpressionRecognizer

# # 画像（OpenCV形式）の読み込み例
# # image = cv2.imread("デバック用_angryが出力されるはず.jpg")
# # image = cv2.imread("デバック用_happyと出力されるはず.jpg")
# recognizer = ExpressionRecognizer(mtcnn=True)

# # start_time = time.perf_counter()# 認識処理前に開始時刻を記録

# json_result = recognizer.recognize(image)

# # 認識処理後に終了時刻を記録
# # end_time = time.perf_counter()
# # processing_time = end_time - start_time
# # print(f"処理時間: {processing_time:.3f}秒")

# print(json_result)



# カメラから取得したフレームを入力データとして使用する場合

#!/usr/bin/env python3
import cv2
from expression_recognition import ExpressionRecognizer

def main():
    # カメラ（デフォルトカメラ:0）をオープン
    cap = cv2.VideoCapture(0)# 環境によってはカメラ番号を変更する必要があるかもしれません
    if not cap.isOpened():
        print("カメラが開けませんでした。")
        return

    # ExpressionRecognizer インスタンスを生成（MTCNN を使用）
    recognizer = ExpressionRecognizer(mtcnn=True)

    print("リアルタイム表情認識を開始します。終了するには 'q' キーを押してください。")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームの取得に失敗しました。")
            break

        # カメラから取得したフレームをそのまま recognizer.recognize() に渡す
        json_result = recognizer.recognize(frame)
        print(json_result)

        # 'q' キーでループ終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

if __name__ == "__main__":
    main()
