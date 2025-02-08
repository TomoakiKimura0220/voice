# emotion_recognition.py
import numpy as np
from tensorflow.keras.models import load_model

# 学習済みモデルファイル (同じディレクトリ内の emotion_model.h5 を使用)
MODEL_PATH = 'emotion_model.h5'
model = load_model(MODEL_PATH)

# 感情ラベル（モデルの出力に合わせて修正してください）
emotion_labels = ['Angry', 'Sad', 'Happy', 'Surprised']

def predict_emotion(features):
    """
    MFCC の平均特徴量 (shape=(13,)) を入力として感情を予測する関数
    
    :param features: numpy 配列, shape=(13,)
    :return: 感情ラベル (文字列)
    """
    # モデルは (1, 13) の入力を想定していると仮定
    input_features = features.reshape(1, -1)
    prediction = model.predict(input_features)
    # 最大の確率のインデックスを取得
    predicted_index = np.argmax(prediction)
    return emotion_labels[predicted_index]
