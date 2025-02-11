# 表情判定モジュール

このプロジェクトは、**fer** ライブラリと MTCNN を用いて、画像またはカメラから取得した映像に対してリアルタイムで顔の表情認識を行うプログラムです。  
検出された複数の顔の中から、画像の中心に最も近い顔の認識結果を JSON 形式で出力するように設計しています。

## 特徴

- **入力データ:**  
  - 画像ファイル（OpenCV形式で読み込んだ画像）  
  - カメラから取得したフレーム

- **認識結果:**  
  JSON 形式の出力で、認識結果には以下の情報が含まれます。  
  - 検出された顔のバウンディングボックス (`box`)
  - 各感情のスコア（`all_emotions`）
  - 最もスコアが高い感情 (`dominant_emotion`) とそのスコア (`score`)

```json
{
  "face": {
    "box": [
      252,
      72,
      165,
      215
    ],
    "dominant_emotion": "angry",
    "score": 0.39,
    "all_emotions": {
      "angry": 0.39,
      "disgust": 0.0,
      "fear": 0.17,
      "happy": 0.0,
      "sad": 0.21,
      "surprise": 0.09,
      "neutral": 0.14
    }
  }
}
```

- **オプション:**  
  - MTCNN を使用するか否かを選択可能（初期設定は `mtcnn=True`）
  - 複数の顔が検出された場合、画像中心に最も近い顔のみを対象にする

## ファイル構成

```bash
.
├── README.md                        # このドキュメント
├── expression_recognition.py        # 表情認識をモジュール化したファイル  
│                                     # → ExpressionRecognizer クラスを定義  
├── main.py                          # モジュールを呼び出して実際に認識を実行するファイル  
│                                     #   ・画像ファイルからの入力例（コメントアウト済み）  
│                                     #   ・カメラからの入力例（リアルタイム認識）
├── requirements.txt                 # 依存パッケージ一覧 (opencv-python, fer, tensorflow など)
└── test_expression_recognition.py   # テスト用ファイル（このファイルだけで実行可能）
```

## セットアップ

1. **仮想環境の作成と有効化**  
    プロジェクトルートで以下を実行してください。
    
    ```bash
    python -m venv venv
    source venv/bin/activate   # Windows の場合: venv\Scripts\activate
    ```
    
2. **依存パッケージのインストール**  
    `requirements.txt` に記載されたパッケージをインストールします。
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ※ fer の動作には TensorFlow（または tensorflow-cpu）や moviepy（fer は moviepy==1.0.3 を前提）などのパッケージも必要です。  
    必要に応じてバージョン調整してください。
    

## 使用方法

### 画像ファイルから認識する場合

`main.py` 内のコメントアウトされたコード部分を有効にし、画像ファイルを読み込んで認識を実行します。

例：

```python
import cv2
from expression_recognition import ExpressionRecognizer

# 画像ファイルを入力データとして使用する例
image = cv2.imread("test_pict_path")
recognizer = ExpressionRecognizer(mtcnn=True)
json_result = recognizer.recognize(image)
print(json_result)
```

### カメラから取得したフレームで認識する場合

`main.py` はデフォルトでカメラ入力からリアルタイム表情認識を実行するようになっています。  
プログラムを実行すると、カメラからのフレームが認識され、結果がコンソールに JSON 形式で出力されます。

実行方法：

```bash
python main.py
```

終了するには、ウィンドウ上で `q` キーを押してください。

### テスト用ファイル

`test_expression_recognition.py` は、カメラから取得した映像に対して直接 fer を用いた認識処理を実行するテスト用のプログラムです。  
個別に実行することで動作確認が可能です。

実行方法：

```bash
python test_expression_recognition.py
```

## 処理時間の測定

呼び出し側のプログラムでは、処理の実行時間を `time.perf_counter()` を用いて計測することも可能です。  
その場合、`recognizer.recognize(frame)` の呼び出し前後で時刻を取得し、差分を出力してください。

例（呼び出し側のコード内）：

```python
import time
start_time = time.perf_counter()
json_result = recognizer.recognize(frame)
end_time = time.perf_counter()
print(f"処理時間: {end_time - start_time:.3f}秒")
```

## 注意事項

* **MTCNN の利用:**  
    `ExpressionRecognizer` クラスの初期化時に `mtcnn=True` を指定すると、MTCNN を用いて顔検出を行います。  
    必要に応じて `mtcnn=False` に切り替えて別の検出手法を利用できます。
    
* **複数顔の処理:**  
    画像内に複数の顔が検出された場合、画像中心に最も近い顔のみの結果を返すようにしています。 
