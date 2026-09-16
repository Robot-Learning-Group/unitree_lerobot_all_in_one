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
  --dataset.repo_id=local/head_only_v1 \
  --dataset.root=/workspace/unitree_lerobot/data/local/head_only_v1 \
  --dataset.video_backend=torchcodec \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --batch_size=32 --num_workers=4 --steps=100000 \
  --save_freq=10000 --eval_freq=0 \
  --output_dir=outputs/train/act_head_only
```

## GR00Tの学習

```bash
# 再学習時は未使用のoutput_dirを指定
docker compose -f docker/compose.yaml run --rm lerobot lerobot-train \
  --dataset.repo_id=local/head_only_v1 \
  --dataset.root=/workspace/unitree_lerobot/data/local/head_only_v1 \
  --dataset.video_backend=torchcodec \
  --policy.type=groot \
  --policy.base_model_path=nvidia/GR00T-N1.5-3B \
  --policy.tune_diffusion_model=false \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --batch_size=32 --num_workers=4 --steps=10000 \
  --save_freq=1000 --eval_freq=0 \
  --output_dir=outputs/train/groot_head_only
```

## 推論

```bash
# GR00Tの場合はact_head_onlyをgroot_head_onlyに変更
# データセットで推論：起動後にsを入力
# 結果：unitree_lerobot/figure.png
docker compose -f docker/compose.yaml run --rm lerobot \
  python -m unitree_lerobot.eval_robot.eval_g1_dataset \
    --policy.path=outputs/train/act_head_only/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --repo_id=local/head_only_v1 \
    --send_real_robot=false \
    --visualization=false

# 実機推論：192.168.123.164:55555でヘッド単眼640×480の画像配信を起動
# 起動時に実機へ接続し、s入力で動作開始
docker compose -f docker/compose.yaml run --rm lerobot \
  python -m unitree_lerobot.eval_robot.eval_g1 \
    --policy.path=outputs/train/act_head_only/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --repo_id=local/head_only_v1 \
    --arm=G1_29 --ee=dex3 --frequency=30 \
    --visualization=false
```
