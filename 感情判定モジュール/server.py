#!/usr/bin/env python3
import asyncio
import websockets
import json
from emotion_recognition import EmotionRecognizer

# グローバルにインスタンスを作成（接続ごとに作成すると時間がかかるため）
recognizer = EmotionRecognizer(window_duration=3.0)

async def handle_connection(websocket, path):
    """
    クライアントから接続があるごとに呼び出されるコールバック関数
    """
    print("クライアントが接続しました")
    try:
        while True:
            # マイクから音声を録音し、感情判定を実行する
            result = recognizer.detect_emotion()
            # 結果の辞書を JSON 文字列に変換（ensure_ascii=False で日本語もそのまま表示）
            json_string = json.dumps(result, ensure_ascii=False)
            await websocket.send(json_string)
            # detect_emotion() は録音・解析に3秒かかるため、ここで追加の待機は不要
    except websockets.exceptions.ConnectionClosed:
        print("クライアントが切断されました")

async def main():
    # localhost の 8765 ポートで WebSocket サーバーを起動
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("WebSocket サーバーが起動しました。クライアントの接続を待っています...")
        await asyncio.Future()  # 永続的に実行

if __name__ == "__main__":
    asyncio.run(main())
