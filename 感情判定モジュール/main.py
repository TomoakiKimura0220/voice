#!/usr/bin/env python3
from emotion_recognition import EmotionRecognizer

def main():
    recognizer = EmotionRecognizer(window_duration=1.0)  # ここで録音窓の長さを指定（秒）
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            result = recognizer.detect_emotion()
            print(result)
            print("【結果】")
            print(f"  録音窓内の総フレーム数: {result['frame_count']}")
            print(f"  中立のカウント: {result['neu_cnt']}")
            print(f"  喜びのカウント: {result['hap_cnt']}")
            print(f"  悲しみのカウント: {result['sad_cnt']}")
            print(f"  怒りのカウント: {result['ang_cnt']}")
            print(f"  多数決の結果: {result['emotion']}")
            print(f"  平均信頼度スコア: {result['average_score']:.2f}\n")
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()
