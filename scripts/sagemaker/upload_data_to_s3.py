"""Upload the SageMaker training channels (libero / base_ckpt) to S3.

Run this once per region before submitting a training job. Idempotent —
re-running only uploads new/changed files (uses `aws s3 sync`).

Reads target S3 URIs from scripts/sagemaker/config.yaml. Override locations
on the CLI:
    python scripts/sagemaker/upload_data_to_s3.py \\
        --lerobot-root ~/.cache/huggingface/lerobot/physical-intelligence/libero \\
        --base-ckpt ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch \\
        --profile sagemaker

Layout that ends up in S3 (matches sm_entrypoint.sh expectations):

    {channels.libero.s3_uri}/{data,meta}/...
    {channels.base_ckpt.s3_uri}/model.safetensors

Mounted in the container at:
    /opt/ml/input/data/libero/{data,meta}/...
    /opt/ml/input/data/base_ckpt/model.safetensors

sm_entrypoint.sh symlinks /opt/ml/input/data/libero -> /tmp/lerobot_root/
physical-intelligence/libero/ and exports HF_LEROBOT_HOME=/tmp/lerobot_root,
so `LeRobotDataset(repo_id="physical-intelligence/libero")` resolves to the
FastFile-mounted parquet tree without any HF Hub fetch.

Norm-stats assets (~20 KB) are baked into the Docker image — no upload.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "scripts" / "sagemaker" / "config.yaml"


def aws_sync(
    src: Path,
    dst: str,
    profile: str,
    region: str,
    *,
    exclude: list[str] | None = None,
    include: list[str] | None = None,
) -> None:
    if not src.exists():
        raise SystemExit(f"Source does not exist: {src}")
    cmd = [
        "aws", "--profile", profile, "--region", region,
        "s3", "sync", str(src), dst,
        "--no-progress",
    ]
    # Order matters in `aws s3 sync`: filters apply in sequence. Exclude first,
    # then re-include the patterns we want.
    for pat in exclude or []:
        cmd += ["--exclude", pat]
    for pat in include or []:
        cmd += ["--include", pat]
    print(f"=> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--profile", default=None, help="AWS profile (defaults to config.yaml aws.profile)")
    p.add_argument("--region", default=None, help="AWS region (defaults to config.yaml aws.region)")
    p.add_argument("--lerobot-root", default="~/.cache/huggingface/lerobot/physical-intelligence/libero",
                   help="Local LeRobot dataset root for physical-intelligence/libero. "
                        "Uploaded as-is — the {data,meta}/ tree lands at the channel root.")
    p.add_argument("--base-ckpt", default="~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch",
                   help="Local pi05_base_pytorch dir (model.safetensors lives here).")
    p.add_argument("--channels", nargs="*",
                   choices=["libero", "base_ckpt", "base_jax", "libero_pytorch"], default=None,
                   help="Subset of targets to upload. Default: training channels only "
                        "(libero, base_ckpt). Opt-ins: `base_jax` mirrors the JAX "
                        "pi05_base orbax checkpoint; `libero_pytorch` mirrors the "
                        "released pi05_libero PyTorch ckpt.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    profile = args.profile or cfg["aws"]["profile"]
    region = args.region or cfg["aws"]["region"]
    channels = cfg["data"]["channels"]
    uploads = cfg["data"].get("uploads", {})

    # Default to training channels only; base_jax must be opted into explicitly
    # (it's 12 GB and not consumed by the running job — it's an upload-only mirror).
    todo = args.channels or list(channels)

    if shutil.which("aws") is None:
        raise SystemExit("aws CLI not on PATH. Install awscli first.")

    if "libero" in todo:
        # LeRobot stores datasets at $HF_LEROBOT_HOME/<repo_id>/{data,meta,...}.
        # We upload that tree as-is to the channel root; on the box,
        # sm_entrypoint.sh symlinks the channel mount to physical-intelligence/
        # libero/ under a writable LeRobot home so LeRobotDataset finds it.
        lerobot_src = Path(args.lerobot_root).expanduser()
        if not (lerobot_src / "meta" / "info.json").exists():
            raise SystemExit(
                f"{lerobot_src}/meta/info.json missing — is the LeRobot LIBERO dataset cached locally? "
                "Run: python -c 'from huggingface_hub import snapshot_download; "
                "snapshot_download(\"physical-intelligence/libero\", repo_type=\"dataset\", "
                "local_dir=\"~/.cache/huggingface/lerobot/physical-intelligence/libero\")'"
            )
        aws_sync(
            lerobot_src, channels["libero"]["s3_uri"], profile, region,
            # Skip LeRobot's local re-cache scratch dir if present.
            exclude=[".cache/*", ".cache/**/*"],
        )

    if "base_ckpt" in todo:
        base_src = Path(args.base_ckpt).expanduser().resolve()
        if not (base_src / "model.safetensors").exists():
            raise SystemExit(f"{base_src}/model.safetensors not found — run examples/convert_jax_model_to_pytorch.py first.")
        aws_sync(base_src, channels["base_ckpt"]["s3_uri"], profile, region)

    if "base_jax" in todo:
        if "base_jax" not in uploads:
            raise SystemExit("config.yaml is missing data.uploads.base_jax. Re-pull the latest config.yaml.")
        spec = uploads["base_jax"]
        jax_src = Path(spec["local_path"]).expanduser()
        if not (jax_src / "params").is_dir():
            raise SystemExit(
                f"{jax_src}/params not found — fetch the JAX base ckpt first:\n"
                "  uv run python -c \"from openpi.shared import download; "
                "print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base'))\""
            )
        aws_sync(jax_src, spec["s3_uri"], profile, region)

    if "libero_pytorch" in todo:
        if "libero_pytorch" not in uploads:
            raise SystemExit("config.yaml is missing data.uploads.libero_pytorch.")
        spec = uploads["libero_pytorch"]
        src = Path(spec["local_path"]).expanduser()
        if not (src / "model.safetensors").exists():
            raise SystemExit(f"{src}/model.safetensors not found — fetch via openpi.shared.download or examples/convert_jax_model_to_pytorch.py.")
        aws_sync(src, spec["s3_uri"], profile, region)

    print("\nDone.")
    for ch in todo:
        if ch in channels:
            print(f"  {ch} (training channel): {channels[ch]['s3_uri']}")
        elif ch in uploads:
            print(f"  {ch} (upload mirror):    {uploads[ch]['s3_uri']}")


if __name__ == "__main__":
    main()
