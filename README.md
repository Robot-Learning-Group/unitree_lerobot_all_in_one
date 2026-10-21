# Unitree LeRobot all in one

## PCのセットアップ（初回のみ）

学習・推論用PC向け。Ubuntu 20.04／22.04／24.04（x86_64）とNVIDIA GPUが前提。

```bash
sudo apt update && sudo apt install -y git git-lfs
git lfs install
git clone --recursive https://github.com/Robot-Learning-Group/unitree_lerobot_all_in_one.git
cd unitree_lerobot_all_in_one

# NVIDIAドライバ580 open・CUDA Toolkit 12.8・Docker・NVIDIA Container Toolkitを導入
# ドライバ未導入の場合はドライバだけ入れて終了するので、再起動後にもう一度実行
sudo bash scripts/install_docker_nvidia.sh
```

完了後、ログアウトして再ログインする（sudoなしでdockerを使うため）。

## ビルド

```bash
# リポジトリのルートで実行（以下も同じ）
git submodule update --init --recursive
docker compose -f docker/compose.yaml build
```

## コンフィグの設定

```bash
# cameras・camera_to_image_keyを編集し、ROBOT_CONFIGSに登録
# 以下は登録済みのG1_DEX3_HEAD_ONLY_CONFIG（Dex3・ヘッド画像color_0のみ）を使用
nano unitree_lerobot/unitree_lerobot/utils/constants.py
```

右腕だけ使いたいなど、特殊な構成を使う場合は、`unitree_lerobot/unitree_lerobot/utils/constants.py` に `RobotConfig` を追加し、`ROBOT_CONFIGS` に名前を登録します。

- `motors`：使用する関節名を、下記のデータ取得順に並べる。
- `json_state_data_name`／`json_action_data_name`：使用する状態・指令の項目（例：`right_arm.qpos`、`right_ee.qpos`）。
- `cameras`／`camera_to_image_key`：使用する画像名と、元データの `color_0` などからの対応。

登録名を変換時の `--robot-type`、実機推論時の `--robot_type` に指定します。学習は変換後のデータセットを使います。現在の設定指定による実機推論は、G1の左右の腕・ハンドの `qpos` に対応しています。ハンドの種類は `--ee`、画像の配信ポートは必要に応じて `--camera_ports` で指定します。

## データセットの変換

```bash
# /絶対パス/data/pick_cube は episode_XXXX/data.json を含む対象タスクのフォルダ
# 出力：unitree_lerobot/data/local/head_only_v1/（同名の既存出力は削除される）
docker compose -f docker/compose.yaml run --rm \
  --volume "/絶対パス/data/データセット名:/rawdata/データセット名:ro" \
  lerobot python -m unitree_lerobot.utils.convert_unitree_json_to_lerobot \
    --raw-dir /rawdata \
    --repo-id local/データセット名 \
    --robot-type G1_DEX3_HEAD_ONLY_CONFIG \
    --dataset-config.image-writer-processes 0 \
    --dataset-config.image-writer-threads 4 \
    --dataset-config.video-backend torchcodec
```
# 学習
## ACTの学習

```bash
# 再学習時は未使用のoutput_dirを指定
docker compose -f docker/compose.yaml run --rm lerobot lerobot-train \
  --dataset.repo_id=local/データセット名 \
  --dataset.root=/workspace/unitree_lerobot/data/local/データセット名 \
  --dataset.video_backend=torchcodec \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --batch_size=32 --num_workers=4 --steps=100000 \
  --save_freq=10000 --eval_freq=0 \
  --output_dir=outputs/train/出力フォルダ名 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50
```
## GR00Tの学習

画像のリサイズ・正規化も `--policy.device` で指定したデバイスで実行します。`cuda` ならGPUを使います。変更は次回の学習起動から反映されます。

```bash
# 再学習時は未使用のoutput_dirを指定
docker compose -f docker/compose.yaml run --rm lerobot lerobot-train \
  --dataset.repo_id=local/データセット名 \
  --dataset.root=/workspace/unitree_lerobot/data/local/データセット名 \
  --dataset.video_backend=torchcodec \
  --policy.type=groot \
  --policy.base_model_path=nvidia/GR00T-N1.5-3B \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --batch_size=32 --num_workers=4 --steps=10000 \
  --save_freq=1000 --eval_freq=0 \
  --output_dir=outputs/train/出力フォルダ名
```
## 学習時に設定する主要なパラメータ
- steps \
  何ステップ学習するか
- batch_size \
  1ステップの学習で扱うサンプル数
- policy.chunk_size \
  ポリシーが一回の推論で行動を何フレーム先まで予測するか. 30ぐらいが良い？
- policy.n_action_steps \
  1回の予測で出力した行動のうち、何ステップ分を実際の行動として実行するか。policy.chunk_size >= policy.n_action_stepsである必要がある。
- save_freq \
  何ステップごとにモデルの重みを保存するか
- wandb.enable \
  trueにするとwandbから学習曲線等が確認できる。wandbのアカウント、APIキーが必要

# 推論

```bash
# データセットで推論：起動後にsを入力
# 結果：unitree_lerobot/figure.png
docker compose -f docker/compose.yaml run --rm lerobot \
  python -m unitree_lerobot.eval_robot.eval_g1_dataset \
    --policy.path=学習したモデル名 （例: outputs/train/board_insert_20260916/checkpoints/100000/pretrained_model） \
    --policy.device=cuda \
    --repo_id=local/データセット名 \
    --send_real_robot=false \
    --visualization=false

# 起動時に実機へ接続し、s入力で動作開始
docker compose -f docker/compose.yaml run --rm lerobot \
  python -m unitree_lerobot.eval_robot.eval_g1 \
    --policy.path=学習したモデル名 （例: outputs/train/board_insert_20260916/checkpoints/100000/pretrained_model）\
    --policy.device=cuda \
    --repo_id=local/データセット名 \
    --image_host=192.168.123.164 --image_port=55555 \
    --arm=G1_29 --ee=dex3 --frequency=30 \
    --visualization=false \
    # 必要に応じて以下の平滑化は削除可能
    --policy.n_action_steps=1 \
    --policy.temporal_ensemble_coeff=-0.01 # ACTのみのパラメータ．これを設定する場合，n_action_steps=1にする
```
ロボットコンフィグを指定する場合は、上記の実機推論コマンドに以下を追加します。

```bash
--robot_type=コンフィグ名 \
--initial_pose_episode=/workspace/datasets/task/episode_0001
```

`robot_type` は変換・学習に使った構成に合わせます。`initial_pose_episode` はコンテナ内のUnitree形式のエピソードフォルダです。`s` 入力で初期姿勢へ移動し、設定された関節で推論を開始します。対象外の腕・使用中ハンドの関節は初期姿勢を保持します。ハンドを含む設定では `--ee` も種類に合わせます。

## 推論時に設定する主要なパラメータ
- image_host \
  カメラサーバーが立ってるパソコンのIPアドレス
- policy.n_action_steps \
  1回の予測で出力した行動のうち、何ステップ分を実際の行動として実行するか。学習時に設定したpolicy.chunk_sizeに対して、 policy.chunk_size >= policy.n_action_stepsである必要がある。
- policy.temporal_ensemble_coeff (ACTのみ) \
  行動の平滑化の係数。-0.02〜0.02あたりで調べるのが良い？マイナスにすると、最新の予測を重視、プラスにすると、過去の予測を重視

# 動かない関節の正規化がポリシーの性能を悪化させる問題について
unitree_lerobot/unitree_lerobot/lerobot/src/lerobot/processor/hand_joint_limits.pyが新しく追加した正規化処理本体。既定でこれを使うようになっている。
従来の正規化を使用する場合は、学習のコマンドで以下を指定する。
```
--policy.hand_joint_limit_normalization=false
```
