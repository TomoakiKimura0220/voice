# 感情判定モジュール

このプロジェクトは、[SpeechBrain](https://speechbrain.github.io) の事前学習済みモデルを利用して、マイクから入力された音声に対してリアルタイムで感情判定を行うシステムです。  
録音した音声を一定のウィンドウ（例：3秒）ごとに解析し、検出された感情ラベル（例：中立、喜び、悲しみ、怒りなど）とその信頼度スコアを表示します。

## ファイル構造
```
.
├── README.md
├── emotion_recognition.py
├── main.py          ← （必要に応じてローカルで実行するエントリーポイント）
├── server.py        ← WebSocket サーバーとして感情判定結果を送信するファイル
└── requirements.txt
```

## 対応OS

- **macOS**
- **Windows**

※ 本プロジェクトは Python 3.9～3.12 で動作します。(SpeechBrainの対応バージョン)

## 環境構築手順

### 1. リポジトリの取得
GitHub からクローンするか、ZIP ファイルとしてダウンロードしてください。

### 2. Python 仮想環境の作成とアクティベート

#### macOS / Linux の場合
ターミナルで以下のコマンドを実行してください。
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows の場合
コマンドプロンプトまたは PowerShell で以下のコマンドを実行してください。
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. 依存パッケージのインストール
仮想環境がアクティブな状態で、以下のコマンドを実行してください。
```bash
pip install -r requirements.txt
```

### 4. プログラムの実行
仮想環境がアクティブな状態で、以下のコマンドを実行してください。
```bash
python main.py
```

実行すると、マイクからの音声入力を一定の時間窓（例: 3秒）ごとに録音し、感情判定結果がコンソールに表示されます。


## 出力形式（JSON形式）

`detect_emotion()` メソッドの実行結果は、以下の形式の辞書として出力され、JSON に変換して利用できます。

```json
{
  "emotion": "喜び",              // 多数決で選ばれた感情（日本語）
  "neu_cnt": 10,                // "neu"（中立）の出現数
  "hap_cnt": 35,                // "hap"（喜び）の出現数
  "sad_cnt": 0,                 // "sad"（悲しみ）の出現数
  "ang_cnt": 0,                 // "ang"（怒り）の出現数
  "average_score": 6.15,        // 多数決対象の感情の平均信頼度スコア
  "frame_count": 49             // 録音窓内の総フレーム数
}
```

この形式で出力されたデータを、WebSocket 経由で送信したり、他のシステムでパースして利用することが可能です。


## プロジェクトの内容

* **emotion_recognition.py**
    * `EmotionRecognizer` クラスを実装しています。
    * マイクからの録音、音声の前処理、SpeechBrain の事前学習済みモデルを用いた感情判定、結果の集約（多数決、平均スコア計算）などの処理を行います。
* **main.py**
    * `EmotionRecognizer` クラスを利用して、リアルタイムで感情認識を実行するエントリーポイントです。
    * 結果は日本語で整形され、コンソールに出力されます。

## 注意事項
* **初回実行時のモデルダウンロード**  
    初回実行時には、SpeechBrain の事前学習済みモデルが Hugging Face Hub からダウンロードされ、`pretrained_models` フォルダにキャッシュされます。ダウンロードには数分かかる場合があります。
    
* **音声入力の環境**  
    マイクの品質や周囲のノイズにより、感情判定の結果が変動することがあります。必要に応じて、録音環境の調整を行ってください。
    
* **ライセンスについて**  
    このプロジェクトは、SpeechBrain などのオープンソースライブラリに依存しています。商用利用の場合は各ライブラリのライセンス条項を必ず確認してください。

## 参考リンク
* SpeechBrain 公式ドキュメント
* [Hugging Face Hub - SpeechBrain モデル](https://huggingface.co/speechbrain)
* [GitHub Issue #2457](https://github.com/speechbrain/speechbrain/issues/2457)
