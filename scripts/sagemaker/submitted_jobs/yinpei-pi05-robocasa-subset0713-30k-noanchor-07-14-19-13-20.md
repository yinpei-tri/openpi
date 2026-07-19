# yinpei-pi05-robocasa-subset0713-30k-noanchor-07-14-19-13-20

- **submitted:** 2026-07-14 19:13:21  (git `68f02dc`)
- **queue:** fss-cv-wfm-spot-p5-48xlarge-us-west-2  (priority 200)
- **instance:** 1x p5, EBS 300GB, max 14d
- **image:** `124224456861.dkr.ecr.us-west-2.amazonaws.com/yinpeidai-openpi-train-jax:20260714-191258`
- **config:** pi05_robocasa_system1  |  **exp_name:** subset0713_30k_noanchor
- **train_args:** `--fsdp-devices=8 --batch-size=128 --ema-decay=0.9999 --num-train-steps=30000 --save-interval=10000 --keep-period=10000 --lr-schedule.warmup-steps=1000 --lr-schedule.peak-lr=1e-4 --lr-schedule.decay-steps=30000 --lr-schedule.decay-lr=1e-5 --model.no-use-anchor-images --data.no-use-anchor-images --data.no-include-anchor-state --project-name=openpi-robocasa-system1`
- **data_loading:** fastfile (FastFile FUSE mount)
- **wandb:** project `openpi-robocasa-system1`  (run id -> `s3://tri-ml-datasets-uw2/yinpeidai/openpi/checkpoints/yinpei-pi05-robocasa-subset0713-30k-noanchor-07-14-19-13-20/pi05_robocasa_system1/subset0713_30k_noanchor/wandb_id.txt`)
- **data channels:**
    - robocasa: s3://tri-ml-datasets-uw2/yinpeidai/robocasa_final_training_shard/system1_subset_0713/ (FastFile)
    - base_ckpt: s3://tri-ml-datasets-uw2/yinpeidai/openpi/released_ckpt/openpi-assets/checkpoints/pi05_base/ (FastFile)
- **norm_stats:** sha256 `e319bb2cc799965e` (state dim 28, actions dim 11; full copy: `yinpei-pi05-robocasa-subset0713-30k-noanchor-07-14-19-13-20.norm_stats.json`)
- **checkpoints:** s3://tri-ml-datasets-uw2/yinpeidai/openpi/checkpoints/yinpei-pi05-robocasa-subset0713-30k-noanchor-07-14-19-13-20/pi05_robocasa_system1/subset0713_30k_noanchor/
- **download ckpts:** `aws s3 sync s3://tri-ml-datasets-uw2/yinpeidai/openpi/checkpoints/yinpei-pi05-robocasa-subset0713-30k-noanchor-07-14-19-13-20/pi05_robocasa_system1/subset0713_30k_noanchor/ ./checkpoints/pi05_robocasa_system1/subset0713_30k_noanchor/ --profile sagemaker`
