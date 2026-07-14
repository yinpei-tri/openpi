# Submitted-job records

One file per SageMaker submission, written automatically by `launch.py` at submit time
(both `<job_name>.json` for tooling and `<job_name>.md` for reading). This is the durable
record so you never have to reconstruct, months later, which job used which data / image /
config or where its outputs live.

Each record captures:
- **job_name** — the SageMaker job id (also the CloudWatch `AWSBatch<job_name>...` stream prefix)
- **git_commit** — the repo state the image was built from
- **image_uri** — the exact immutable ECR tag baked for this run
- **train_config + exp_name + train_args** — the full trainer invocation
- **data_channels** — each S3 dataset URI + input mode (and `download_to_local` if the
  entrypoint `aws s3 sync`ed the data to local EBS instead of FastFile)
- **wandb_project** + `wandb_id_file` — the wandb run id is generated in-container and
  written to `<ckpt_dir>/wandb_id.txt`; this points you at it
- **checkpoint_s3_dir** + **ckpt_download_cmd** — where the checkpoints land + how to pull them
- queue / priority / instance / EBS / max_run_days

## Note on norm_stats (the one thing NOT fully captured here)
`norm_stats.json` is baked into the image from `assets/<config>/<asset_id>/` at build time
and is gitignored + mutated in place per dataset. The record pins the **image tag** and
**git commit**, but to know the exact norm_stats bytes for a job, keep the dataset's
`norm_stats.json` alongside its shards in S3 (the producer already does this) — the
authoritative copy is `s3://<dataset>/norm_stats.json`, matched to `data_channels.robocasa`.
