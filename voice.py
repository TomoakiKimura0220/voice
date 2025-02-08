# 一秒分の音声データから音圧と音量（dB）を計算するプログラム
# pyaudio と numpy ライブラリを使用しています
# PyAudio・・・ver.0.2.14
# NumPy・・・ver.1.26.4

import pyaudio     # マイクなどのオーディオ入力を扱うライブラリ
import numpy as np # 数値計算用ライブラリ

# --- 音声取得の設定パラメータ ---
CHUNK = 1024             # 1チャンクあたりのサンプル数
FORMAT = pyaudio.paInt16 # 16ビット整数形式の音声データ
CHANNELS = 1             # モノラル録音
RATE = 44100             # サンプリングレート (Hz)、1秒間に取得するサンプル数

# --- PyAudio の初期化と入力ストリームの設定 ---
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("録音中...（Ctrl+C で終了）")

try:
    while True:
        # --- 1秒間分の音声データを取得する ---
        frames = []  # 1秒分のチャンクデータを格納するリスト
        
        # 1秒に必要なチャンク数を計算（小数点以下は切り捨て）
        num_chunks = int(RATE / CHUNK)
        
        # 1秒間分のデータを連続で読み込む
        for i in range(num_chunks):
            data = stream.read(CHUNK)
            # 取得したバイナリデータを int16 の NumPy 配列に変換し、計算のために float32 にキャスト
            frames.append(np.frombuffer(data, dtype=np.int16).astype(np.float32))
        
        # 複数のチャンクを連結して、1秒分の音声データを作成する
        audio_data = np.concatenate(frames)
        
        # --- RMS（Root Mean Square: 二乗平均平方根）の計算 ---
        # RMS は信号の大きさ（パワー）の指標として使われる
        rms = np.sqrt(np.mean(audio_data**2))
        
        # --- dB 値への変換 ---
        # 一般的に、dB 値は 20 * log10(振幅) で求める
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            db = -np.inf  # rms が 0 の場合は対数が定義できないので -∞ とする
        
        # 1秒間分の結果を出力する
        print(f"1秒間のRMS: {rms:.2f}  |  dB: {db:.2f} dB")
        
except KeyboardInterrupt:
    print("\n録音終了")
finally:
    # --- 後処理 ---
    stream.stop_stream()
    stream.close()
    p.terminate()
