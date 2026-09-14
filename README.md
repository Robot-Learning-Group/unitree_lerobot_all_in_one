# Unitree LeRobot all in one

Unitree公式のソースを変更せずに取り込み、ACT・LeRobot版GR00Tの学習・推論に必要なDocker環境を構築します。今回の対象は環境構築です。公式の実機推論コードの修正は含めません。

## ソース

2026-09-14に公式リポジトリから取得しました。Unitreeの2リポジトリは取得時の既定ブランチ、LeRobotはUnitree公式のgitlinkが指定するコミットです。

| ディレクトリ | 公式取得元 | コミット | ライセンス |
| --- | --- | --- | --- |
| `unitree_lerobot/` | [unitreerobotics/unitree_lerobot](https://github.com/unitreerobotics/unitree_lerobot) | `41c2805742de879ddab2d8d6beaeaf215f876395` | Apache-2.0 |
| `unitree_sdk2_python/` | [unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) | `65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5` | BSD-3-Clause |
| `unitree_lerobot/unitree_lerobot/lerobot/` | [huggingface/lerobot](https://github.com/huggingface/lerobot) | `a5b29d430105f5235eb05bbf2db5a0d747a869d6` | Apache-2.0 |

Unitree LeRobotと同梱LeRobotは、ソース・アセット・ライセンスをこのリポジトリで直接管理します。Git LFSのアセットは実体を取得しています。この2つの取り込み時には上流の `.git`、`.gitmodules`、使用しないDocker関連ファイルを除外しました。SDKは公式リポジトリを参照するsubmoduleで、上記コミットに固定しています。ローカルの他プロジェクトからのコピーはありません。

LeRobotの `tests/artifacts/` にあるテスト用バイナリ45ファイル（safetensors・カメラ画像・RealSense記録）はGit管理から除外し、同ディレクトリをDockerのビルド対象からも除外しています。学習・推論用のコード、テスト用Pythonコード、ロボットのURDF・メッシュは保持しています。上流の資産依存テストを実行する場合は、上記LeRobotコミットからテスト資産を別途取得してください。Dockerの環境検査はこれらに依存しません。

## 起動

Linux x86_64、Docker Compose、NVIDIAドライバ、NVIDIA Container Toolkitを使用します。ホストへのPython・CUDA Toolkitのインストールは不要です。

リポジトリのルートで実行します。

```bash
git submodule update --init --recursive
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml run --rm lerobot
```

コンテナの作業ディレクトリは `/workspace/unitree_lerobot` です。以下のPythonコマンドはこのディレクトリで実行します。シェルを開かず、`docker compose -f docker/compose.yaml run --rm lerobot` の後ろに直接コマンドを付けても実行できます。

環境はPython 3.10、CUDA 12.6.3、PyTorch 2.7.1、torchvision 0.22.1、TorchCodec 0.5、FlashAttention 2.8.3です。Pinocchio 3.3.1（CasADi対応）・FFmpeg 7.1.1はconda-forgeから導入します。Isaac-GR00T本体・シミュレータ・専用IKサービスは含みません。

ソースはホストから `/workspace` にマウントし、3パッケージをeditable installします。Pythonの編集後は実行プロセスを再起動してください。依存・Dockerfileの変更時は再ビルドが必要です。

生成ファイルのUID/GIDはマウントしたリポジトリの所有者から取得します。必要なら明示できます。

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) \
  docker compose -f docker/compose.yaml run --rm lerobot
```

ビルド時の並列コンパイル数は `MAX_JOBS`（既定値2）で変更できます。

## 保存先

| 用途 | ホスト側の場所 | コンテナ側の場所 |
| --- | --- | --- |
| 変換元データ | `unitree_lerobot/raw_data/` | `/workspace/unitree_lerobot/raw_data/` |
| LeRobotデータセット | `unitree_lerobot/data/` | `/workspace/unitree_lerobot/data/` |
| チェックポイント・出力 | `unitree_lerobot/outputs/` | `/workspace/unitree_lerobot/outputs/` |
| HF・Torch等のキャッシュ | `unitree_lerobot/.cache/` | `/workspace/unitree_lerobot/.cache/` |

これらはGit管理とDockerビルド対象から除外し、コンテナ削除後も保持します。`HF_LEROBOT_HOME` は `data/`、`HF_HOME` は `.cache/huggingface/` を指します。Hubの認証が必要な場合はコンテナ内で `hf auth login` を実行します。

## 環境の確認

コンテナ内で公式CLIを確認できます。作業ディレクトリは `/workspace/unitree_lerobot` のままにします。

```bash
lerobot-train --help
python -m unitree_lerobot.utils.convert_unitree_json_to_lerobot --help
python -m unitree_lerobot.eval_robot.eval_g1_dataset --help
python -m unitree_lerobot.eval_robot.eval_g1 --help
```

ビルド時には `docker/smoke_test.py` で次を確認します。モデルのダウンロードや実機への接続は行いません。

- ACT・GR00T・SDK・画像通信・CasADi対応Pinocchioのimport
- FFmpegで生成した短い動画をTorchCodecとDecordでデコード
- G1のURDF読み込み、CasADiモデル作成、重力補償トルク計算
- 学習・データ変換・評価CLIのヘルプ表示
- Pythonの依存整合性

再実行する場合：

```bash
docker compose -f docker/compose.yaml run --rm lerobot python /opt/smoke_test.py
docker compose -f docker/compose.yaml run --rm lerobot \
  python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name()); print(torch.ones(2, device="cuda") + 1)'
```

Decord 0.6.0の公式wheelにはプラットフォームタグの問題があり、`pip check` が `decord 0.6.0 is not supported on this platform` と表示します。[上流の報告](https://github.com/dmlc/decord/issues/366)。環境検査ではこのメッセージだけを許容し、実際の動画デコードも検証します。それ以外の依存不整合はビルドを失敗させます。Decordのコードやメタデータは変更しません。

SDKとLeRobotがそれぞれ宣言する `opencv-python` と `opencv-python-headless` は、公式の依存宣言を変更せずに導入します。

## 学習・推論の入口と今回の範囲

データ形式・学習・評価コマンドは [Unitree公式README](unitree_lerobot/README.md) と [LeRobot README](unitree_lerobot/unitree_lerobot/lerobot/README.md) を参照してください。GPU、データ、チェックポイントが揃った状態での学習・実機動作は、環境構築とは別の確認が必要です。

取得した公式コードには、静的確認で次の不整合があります。今回は修正せず、後続の別コミットで扱います。

- `setup_image_client` が要求する `image_host` が `EvalRealConfig` にない。
- `setup_image_client` はクライアントと設定の組を返すが、`eval_g1` は旧形式の辞書としてアクセスする。
- `process_images_and_observations` は3引数だが、`eval_g1` は7引数で呼び出す。
- 実機・データセット評価で `--root` がデータセット生成に渡されていない。

`eval_g1` は実機用です。公式実装では開始入力の前にロボットコントローラを初期化し、`--send_real_robot=false` を実機通信の停止には使いません。今回の環境検査ではコントローラやDDS participantを作成しません。

## 検証結果

2026-09-14、RTX 4090（24GB）、NVIDIAドライバ565.77のホストで確認しました。

| 検査 | 結果 |
| --- | --- |
| 公式取得元との照合 | テスト資産除外後の直接管理688ファイルと、SDK submodule内の266ファイルが公式ソースと一致 |
| Compose設定・entrypoint構文 | 成功 |
| Dockerイメージのビルド | 成功 |
| ACT・GR00T・SDK・画像通信等のimport | 成功 |
| FFmpeg → TorchCodec／Decordの動画デコード | 成功 |
| CasADi対応Pinocchio・G1のURDFと重力補償計算 | 成功 |
| 学習・変換・実機／データセット評価CLIのヘルプ | 成功 |
| CUDA・FlashAttentionの順伝播／逆伝播 | 成功。PyTorchのattention出力とも比較 |
| マウントした公式ソースでの環境検査 | 成功 |
| ホストのUID/GIDでのファイル作成・別コンテナでの永続化 | 成功 |
| Python依存整合性 | 上記Decordのwheelタグ警告のみ。その他の不整合なし |

学習の実行、学習済みモデルの評価、実機への接続・動作指令は行っていません。公式ソースの不整合を解消したことを示す検査ではありません。
