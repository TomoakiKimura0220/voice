#!/usr/bin/env python3
import speechbrain as sb
import torch
import sounddevice as sd
import numpy as np
from collections import Counter

class EmotionRecognizer:
    def __init__(self,
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                savedir="pretrained_models/emotion-recognition-wav2vec2-IEMOCAP",
                samplerate=16000,
                window_duration=3.0):
        """
        コンストラクタ
        
        Parameters:
            source (str): Hugging Face Hub から取得するモデルのID
            savedir (str): モデルのキャッシュ先ディレクトリ
            samplerate (int): 録音のサンプルレート（Hz）
            window_duration (float): 分析用の録音窓の長さ（秒）
        """
        self.classifier = sb.inference.EncoderClassifier.from_hparams(
            source=source,
            savedir=savedir
        )

        # Monkey-patch: compute_features の呼び出しを extract_features に変更
        if not hasattr(self.classifier.mods, "compute_features"):
            if "wav2vec2" in self.classifier.mods:
                def compute_features_wrapper(wavs):
                    return self.classifier.mods["wav2vec2"].extract_features(wavs)
                self.classifier.mods.compute_features = compute_features_wrapper
            else:
                raise AttributeError("compute_features メソッドを含む適切なモジュールが見つかりません。")

        # Monkey-patch: mean_var_norm をダミー関数に置き換え
        if not hasattr(self.classifier.mods, "mean_var_norm"):
            def mean_var_norm_wrapper(feats, wav_lens):
                return feats
            self.classifier.mods.mean_var_norm = mean_var_norm_wrapper

        # Monkey-patch: embedding_model をダミー関数に置き換え
        if not hasattr(self.classifier.mods, "embedding_model"):
            def embedding_model_wrapper(feats, wav_lens):
                return feats
            self.classifier.mods.embedding_model = embedding_model_wrapper

        # classifier モジュールが見つからない場合、output_mlp または avg_pool から取得
        if not hasattr(self.classifier.mods, "classifier"):
            found_classifier = False
            for key in ["output_mlp", "avg_pool"]:
                if key in self.classifier.mods:
                    self.classifier.mods.classifier = self.classifier.mods[key]
                    found_classifier = True
                    break
            if not found_classifier:
                raise AttributeError("Classifier モジュールが見つかりません。")

        self.samplerate = samplerate
        self.window_duration = window_duration
        self.num_samples = int(self.samplerate * self.window_duration)
        # 感情ラベルの英語から日本語への対応
        self.emotion_map = {
            "neu": "中立",
            "hap": "喜び",
            "sad": "悲しみ",
            "ang": "怒り"  # 他の感情があれば追加
        }

    def detect_emotion(self):
        """
        マイクから録音した音声に対して感情認識を実行し、結果を辞書で返す。
        
        Returns:
            dict: { "emotion": 日本語の感情ラベル,
                    "raw_emotions": 各フレームの予測結果リスト,
                    "average_score": 多数決に対応する平均スコア,
                    "frame_count": 録音窓内の総フレーム数 }
        """
        # 3秒間の録音（blocking call）
        audio = sd.rec(int(self.num_samples), samplerate=self.samplerate, channels=1, dtype='float32')
        sd.wait()  # 録音終了まで待機

        # 取得した音声データは shape (num_samples, 1) なので、1次元に変換し、バッチ次元を追加
        audio_tensor = torch.from_numpy(audio.flatten()).unsqueeze(0)  # shape: [1, num_samples]

        # classify_batch() は (out_prob, score, index, text_lab) を返す
        out_prob, score, index, text_lab = self.classifier.classify_batch(audio_tensor)

        # text_lab[0] は各フレームの予測結果リストと仮定
        predictions = text_lab[0]
        counter = Counter(predictions)
        majority_emotion, count = counter.most_common(1)[0]
        # 英語ラベルを日本語に変換
        majority_emotion_jp = self.emotion_map.get(majority_emotion, majority_emotion)
        # 多数決の結果が出現するフレームのスコア平均を計算
        indices = [i for i, p in enumerate(predictions) if p == majority_emotion]
        if indices:
            majority_scores = score[0][indices]
            avg_score = majority_scores.mean().item()
        else:
            avg_score = float('nan')

        result = {
            "emotion": majority_emotion_jp,
            "raw_emotions": predictions,
            "average_score": avg_score,
            "frame_count": len(predictions)
        }
        return result
