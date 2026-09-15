# Unitree LeRobot all in one

Unitree LeRobotの公式リリース `v3.0` を変更せずに取り込み、ACT・LeRobot版GR00Tの学習・推論に必要なDocker環境を構築します。ACTの短時間学習を確認するための、XR記録データの変換・実行手順も含みます。

## ソース

2026-09-15にUnitree LeRobotを公式タグ `v3.0` に揃えました。LeRobotは同タグのgitlinkが指定するコミットです。SDKは2026-09-14に公式の既定ブランチから取得したコミットを維持しています。

| ディレクトリ | 公式取得元 | コミット | ライセンス |
| --- | --- | --- | --- |
| `unitree_lerobot/` | [unitreerobotics/unitree_lerobot v3.0](https://github.com/unitreerobotics/unitree_lerobot/tree/v3.0) | `66ca89d6fffd919c001e7af06f116dca78e01cf1` | Apache-2.0 |
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

`logging_mp` はDocker側で `0.1.6` に固定します。これは `v3.0` 公開時のバージョンで、公式コードが使う `basic_config`・`get_logger` に対応しています。上流の `pyproject.toml` は変更していません。

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

## XR記録データでACTの学習を確認

以下はリポジトリのルートで実行します。隣の `xr_teleoperate_all_in_one/data/pick cube` のうち、`episode_0001`〜`episode_0008` を使用します。8エピソード・1,431フレーム、30 FPS、640×480の画像3系統、状態・行動はそれぞれ28関節です。`episode_0009`以降は手の関節数が異なるため、このデータセットには混ぜません。

公式のDex3変換設定は4カメラを前提にしています。この記録は3カメラなので、本プロジェクトの [docker/convert_xr_dataset.py](docker/convert_xr_dataset.py) が公式LeRobotのデータセットAPIを使って変換します。`color_0`を頭部、`color_1`を左手首、`color_2`を右手首として保存します。公式ソースと元の記録ファイルは変更せず、カメラや関節の不足分を補完することもありません。

変換元を読み取り専用でマウントします。出力先が既に存在する場合、変換スクリプトは上書きせず終了します。再変換するときは別の `--repo-id` を指定してください。

```bash
docker compose -f docker/compose.yaml run --rm -T \
  --volume "$(realpath ../xr_teleoperate_all_in_one/data):/rawdata:ro" \
  lerobot env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 \
  python /workspace/docker/convert_xr_dataset.py \
    --raw-dir '/rawdata/pick cube' \
    --repo-id local/pick_cube_dex3_v3 \
    --episodes 1 2 3 4 5 6 7 8
```

変換結果は `unitree_lerobot/data/local/pick_cube_dex3_v3/` に保存されます。`meta/conversion_source.json` に元エピソード番号・フレーム数・JSONのSHA-256を記録します。

次に、公式の学習CLIでACTを10ステップ実行します。動作確認用としてバッチサイズを2にし、ResNetの事前学習済み重みは使いません。学習ループ・逆伝播・チェックポイント保存の確認が目的で、ポリシー性能の評価ではありません。

```bash
docker compose -f docker/compose.yaml run --rm -T lerobot \
  env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 \
  lerobot-train \
    --dataset.repo_id=local/pick_cube_dex3_v3 \
    --dataset.root=/workspace/unitree_lerobot/data/local/pick_cube_dex3_v3 \
    --dataset.video_backend=torchcodec \
    --policy.type=act \
    --policy.device=cuda \
    --policy.pretrained_backbone_weights=null \
    --policy.push_to_hub=false \
    --wandb.enable=false \
    --batch_size=2 --num_workers=2 --steps=10 \
    --log_freq=1 --save_freq=10 --eval_freq=0 \
    --output_dir=outputs/train/act_pick_cube_v3_smoke
```

再実行時は `--output_dir` を別名にしてください。データと出力はGit管理・Dockerビルド対象から除外しています。

## 推論について

データ形式・学習・評価コマンド全般は [Unitree公式README](unitree_lerobot/README.md) と [LeRobot README](unitree_lerobot/unitree_lerobot/lerobot/README.md) を参照してください。実機推論とポリシー性能は今回の学習確認の対象外です。

`v3.0`では、mainで見つかった画像クライアントの戻り値・画像処理の引数の不整合はありません。一方、実機・データセット評価の `--root` がデータセット生成に渡されない点は残っています。公式コードは修正していません。

`eval_g1` は実機用です。公式実装では開始入力の前にロボットコントローラを初期化し、`--send_real_robot=false` を実機通信の停止には使いません。今回の環境検査ではコントローラやDDS participantを作成しません。

## 検証結果

2026-09-15、RTX 4090（24GB）、NVIDIAドライバ565.77のホストで `v3.0` を確認しました。

| 検査 | 結果 |
| --- | --- |
| 公式取得元との照合 | Unitree v3.0の201ファイルと同梱LeRobotの464ファイルが公式と一致。SDK submoduleも指定コミットから変更なし |
| Compose設定・entrypoint構文 | 成功 |
| Dockerイメージのビルド | 成功 |
| ACT・GR00T・SDK・画像通信等のimport | 成功 |
| FFmpeg → TorchCodec／Decordの動画デコード | 成功 |
| CasADi対応Pinocchio・G1のURDFと重力補償計算 | 成功 |
| 学習・変換・実機／データセット評価CLIのヘルプ | 成功 |
| CUDA・FlashAttentionの順伝播／逆伝播 | 成功。PyTorchのattention出力とも比較 |
| マウントした公式ソースの実行 | ACT学習・保存モデルの読み込みに成功 |
| ホストのUID/GIDでのファイル作成・別コンテナでの永続化 | データとモデルがUID 1004／GID 1005で保存され、別コンテナから読み込み成功 |
| Python依存整合性 | 上記Decordのwheelタグ警告のみ。その他の不整合なし |
| XRデータ変換 | 8エピソード・1,431フレームをLeRobot dataset v3に変換。全エピソードの先頭・末尾16フレームを動画デコード |
| 変換時の入力確認 | 異なる手の関節数は出力作成前に拒否。既存データセットの上書きも拒否 |
| ACT学習 | 公式CLIがCUDAで10ステップ完走。全ステップでloss・勾配ノルムが有限値 |
| チェックポイント | step 10、モデル・optimizer・前後処理を保存。ACTモデルの再読み込みに成功し、保存テンソル234個がすべて有限値 |

学習時のlossは最初が61.872、最後が31.064でした。これは動作確認の記録で、性能を示す評価値ではありません。

ローカルの成果物（Git管理対象外）：

- データ：`unitree_lerobot/data/local/pick_cube_dex3_v3/`
- モデル：`unitree_lerobot/outputs/train/act_pick_cube_v3_smoke/checkpoints/000010/pretrained_model/`
- ログ：`unitree_lerobot/outputs/train/act_pick_cube_v3_smoke/` 内の `training.log`、`conversion.log`、`checkpoint_check.log`、`docker_build.log`

学習済みポリシーの性能評価、実機への接続・動作指令は行っていません。
