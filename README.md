# Unitree LeRobot all in one

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

```bash
# 再学習時は未使用のoutput_dirを指定
docker compose -f docker/compose.yaml run --rm lerobot lerobot-train \
  --dataset.repo_id=local/データセット名 \
  --dataset.root=/workspace/unitree_lerobot/data/local/データセット名 \
  --dataset.video_backend=torchcodec \
  --policy.type=groot \
  --policy.base_model_path=nvidia/GR00T-N1.5-3B \
  --policy.tune_diffusion_model=false \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --batch_size=32 --num_workers=4 --steps=10000 \
  --save_freq=1000 --eval_freq=0 \
  --output_dir=outputs/train/出力フォルダ名
```

## 推論

```bash
# データセットで推論：起動後にsを入力
# 結果：unitree_lerobot/figure.png
docker compose -f docker/compose.yaml run --rm lerobot \
  python -m unitree_lerobot.eval_robot.eval_g1_dataset \
    --policy.path=outputs/train/出力フォルダ名/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --repo_id=local/データセット名 \
    --send_real_robot=false \
    --visualization=false

# 起動時に実機へ接続し、s入力で動作開始
docker compose -f docker/compose.yaml run --rm lerobot \
  python -m unitree_lerobot.eval_robot.eval_g1 \
    --policy.path=outputs/train/出力フォルダ名/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --repo_id=local/データセット名 \
    --image_host=192.168.123.164 --image_port=55555 \
    --arm=G1_29 --ee=dex3 --frequency=30 \
    --visualization=false
```

### 動かない関節の正規化がポリシーの性能を悪化させる問題について
unitree_lerobot/unitree_lerobot/lerobot/src/lerobot/processor/hand_joint_limits.pyが新しく追加した正規化処理本体。既定でこれを使うようになっている。
従来の正規化を使用する場合は、学習のコマンドで以下を指定する。
```
--policy.hand_joint_limit_normalization=false
```