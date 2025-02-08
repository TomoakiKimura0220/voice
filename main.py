# main.py
import numpy as np
import pyaudio
import librosa
from emotion_recognition import predict_emotion

# オーディオパラメータ
SAMPLE_RATE = 16000      # サンプリングレート (Hz)
CHUNK = 1024             # 1 回の読み込みサイズ（サンプル数）
WINDOW_SECONDS = 1       # 1 回の処理に使う音声の長さ（秒）

# 1 秒間に必要なチャンク数
WINDOW_SIZE = int(SAMPLE_RATE / CHUNK * WINDOW_SECONDS)

def capture_audio():
    """マイクから音声を取得し、1 秒ごとに感情認識を実施する関数"""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("マイク入力を開始します。Ctrl+C で終了。")
    
    try:
        while True:
            frames = []
            # 1 秒分のデータを収集
            for _ in range(WINDOW_SIZE):
                data = stream.read(CHUNK)
                frames.append(data)
            
            # バイト列を数値データに変換
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
            # 正規化（16bit の最大値で割る）
            audio_data = audio_data / 32768.0
            
            # Librosa を使って MFCC 特徴量を抽出（次元数は 13 例）
            mfcc = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=13)
            # 時間軸方向の平均値を取って shape=(13,) にする
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # 学習済みモデルで感情を推論
            emotion = predict_emotion(mfcc_mean)
            print("推定された感情:", emotion)
    
    except KeyboardInterrupt:
        print("\n解析を終了します。")
    
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':
    capture_audio()
