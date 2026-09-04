# Embodied Vision

YOLO + DeepSORTによる人物追跡と、DeepFaceによる登録人物照合を行うローカルサービスです。

## 動作

- YOLOで検出した人物をDeepSORTの`track_id`で追跡します。
- `person_appeared`、`person_disappeared`を`kokomi_kernel`へ送ります。
- 顔照合は追跡ループとは別の1ワーカースレッドで低頻度に実行します。
- 同じ登録人物との一致が連続したときだけ`person_recognized`を送ります。
- 未登録の場合は`person_unknown`を一度だけ送ります。
- 登録時は顔画像ではなく顔特徴ベクトルだけを`data/face_registry.json`へ保存します。

## セットアップ

Python 3.11環境で以下を実行します。

```powershell
pip install -r deepsort-py311requirements.txt
pip install -r face-recognition-requirements.txt
python deepsort.py
```

DeepFace/SFaceのモデルファイルは初回利用時に取得されます。

主な環境変数：

| 変数 | 既定値 | 用途 |
|---|---:|---|
| `VISION_EVENT_SEND_ENABLED` | `true` | Aliceへのイベント送信 |
| `VISION_EVENT_URL` | `http://localhost:3000/yolo_event` | イベント送信先 |
| `VISION_HTTP_HOST` | `127.0.0.1` | HTTP APIの待受アドレス |
| `VISION_HTTP_PORT` | `5000` | HTTP APIの待受ポート |
| `VISION_DISPLAY_ENABLED` | `true` | OpenCVウィンドウ表示 |
| `FACE_RECOGNITION_ENABLED` | `true` | 登録顔照合 |
| `FACE_MODEL_NAME` | `SFace` | DeepFaceの顔認識モデル |
| `FACE_DETECTOR_BACKEND` | `opencv` | 顔検出バックエンド |
| `FACE_MATCH_THRESHOLD` | `0.45` | コサイン距離の一致上限 |
| `FACE_CONFIRMATIONS` | `2` | 認識確定に必要な連続一致数 |
| `FACE_ANALYSIS_INTERVAL_SECONDS` | `1.0` | 同一トラックの顔解析間隔 |
| `FACE_REGISTRY_PATH` | `data/face_registry.json` | 登録データ保存先 |

## HTTP API

- `GET /status`：カメラ・顔認識ワーカーの状態
- `GET /snapshot`：Aliceに渡す生画像
- `GET /snapshot/annotated`：IDと認識名を描画した確認画像
- `GET /tracks`：現在のトラックと認識状態
- `GET /faces`：登録人物一覧。顔特徴ベクトルは返しません
- `POST /faces/enroll`：現在のトラックを登録

登録例：

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:5000/faces/enroll `
  -ContentType application/json `
  -Body '{"track_id":"7","name":"たかん"}'
```

Aliceからは`remember_person`アクションで同じAPIを呼び出せます。顔がまだ正しく取得できていない場合はHTTP 409となり、誤った画像を登録しません。同じ名前で再登録すると、最大5個まで別角度の特徴ベクトルが追加されます。
