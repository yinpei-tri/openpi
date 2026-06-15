"""Submit an openpi pi05_libero PyTorch training job to SageMaker.

All defaults live in scripts/sagemaker/config.yaml. Override on the CLI:
    python scripts/sagemaker/launch.py --instance.type=p5 --training.exp_name=run2

The queue name is constructed as `fss-{queue.name}-{instance}-{region}`,
e.g. `fss-vla-p5-48xlarge-us-west-2`.

The image is built from scripts/sagemaker/train.Dockerfile, with the openpi
source baked in (no SageMaker source_dir upload). The estimator submits the
image with three input channels (libero, base_ckpt, assets) — see config.yaml.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
import yaml
from sagemaker.aws_batch.training_queue import TrainingQueue
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

REPO_ROOT = Path(__file__).resolve().parents[2]

INSTANCE_MAPPER = {
    "p4d": "ml.p4d.24xlarge",
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
    "p5en": "ml.p5en.48xlarge",
    "p6": "ml.p6-b200.48xlarge",
}

QUEUE_SUFFIX = {
    "us-west-2": {
        "ml.p4d.24xlarge": "p4d-24xlarge-us-west-2",
        "ml.p4de.24xlarge": "p4de-24xlarge-us-west-2",
        "ml.p5.48xlarge": "p5-48xlarge-us-west-2",
        "ml.p5en.48xlarge": "p5en-48xlarge-us-west-2",
        "ml.p6-b200.48xlarge": "p6-b200-48xlarge-us-west-2",
    },
}


# -----------------------------------------------------------------------------
# Config loading: YAML + CLI overrides
# -----------------------------------------------------------------------------
def _set_dotted(d: dict, dotted_key: str, value: str) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = _coerce(value, cur.get(keys[-1]))


def _coerce(value: str, existing):
    if isinstance(existing, bool):
        return value.lower() in {"1", "true", "yes", "y"}
    if isinstance(existing, int) and not isinstance(existing, bool):
        return int(value)
    if isinstance(existing, float):
        return float(value)
    return value


def load_config() -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--local", action="store_true", help="Run via SageMaker LocalSession")
    parser.add_argument("--dry-run", action="store_true", help="Build config, skip image build + submit")
    parser.add_argument("--skip-build", action="store_true", help="Use existing :latest image, don't rebuild")
    args, overrides = parser.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for raw in overrides:
        if not raw.startswith("--") or "=" not in raw:
            raise SystemExit(f"Bad override (use --key.path=value): {raw}")
        key, value = raw[2:].split("=", 1)
        _set_dotted(cfg, key, value)

    cfg["_meta"] = {"local": args.local, "dry_run": args.dry_run, "skip_build": args.skip_build}
    return cfg


# -----------------------------------------------------------------------------
# Image build & push
# -----------------------------------------------------------------------------
def run(cmd: str) -> None:
    print(f"=> {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=REPO_ROOT)


def ecr_account(region: str, profile: str) -> str:
    out = subprocess.check_output(
        ["aws", "--region", region, "--profile", profile,
         "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
        text=True,
    ).strip()
    if not out.isdigit():
        raise RuntimeError(f"Invalid account: {out!r}")
    return out


def build_and_push_image(cfg: dict) -> str:
    region = cfg["aws"]["region"]
    profile = cfg["aws"]["profile"]
    repo = cfg["image"]["repo_name"]

    account = ecr_account(region, profile)
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{repo}:latest"
    dockerfile = REPO_ROOT / "scripts" / "sagemaker" / "train.Dockerfile"

    login_dlc = (
        f"aws ecr get-login-password --region {region} --profile {profile} | "
        f"docker login --username AWS --password-stdin "
        f"763104351884.dkr.ecr.{region}.amazonaws.com"
    )
    login_self = (
        f"aws ecr get-login-password --region {region} --profile {profile} | "
        f"docker login --username AWS --password-stdin "
        f"{account}.dkr.ecr.{region}.amazonaws.com"
    )

    run(login_dlc)
    run(
        f"docker build --progress=plain -f {dockerfile} "
        f"--build-arg AWS_REGION={region} -t {repo} ."
    )
    run(f"docker tag {repo} {fullname}")
    run(login_self)
    # Create the ECR repo if it doesn't exist (idempotent).
    run(
        f"aws --region {region} --profile {profile} ecr describe-repositories "
        f"--repository-names {repo} --no-cli-pager > /dev/null 2>&1 || "
        f"aws --region {region} --profile {profile} ecr create-repository "
        f"--repository-name {repo} --no-cli-pager"
    )
    run(f"docker push {fullname}")
    time.sleep(3)
    return fullname


# -----------------------------------------------------------------------------
# Job-name helpers
# -----------------------------------------------------------------------------
def sanitize(name: str) -> str:
    name = name.replace("_", "-").replace(".", "-")
    name = "".join(c if c.isalnum() or c == "-" else "" for c in name)
    return name.strip("-") or "job"


def make_job_name(prefix: str, user: str) -> str:
    base = sanitize(f"{user}-{prefix}" if prefix else user)
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"{base}-{stamp}"[:63].rstrip("-")


# -----------------------------------------------------------------------------
# Secrets / env
# -----------------------------------------------------------------------------
def load_secrets(path: str = "secrets.env") -> dict[str, str]:
    p = REPO_ROOT / "scripts" / "sagemaker" / path
    if not p.exists():
        return {}
    out = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("\"'")
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    local = cfg["_meta"]["local"]
    dry_run = cfg["_meta"]["dry_run"]
    skip_build = cfg["_meta"]["skip_build"]

    role = cfg["aws"]["role_arn"] or os.environ.get("SAGEMAKER_ARN")
    if not role:
        raise SystemExit("Set aws.role_arn in config.yaml or $SAGEMAKER_ARN.")

    short = cfg["instance"]["type"]
    if short not in INSTANCE_MAPPER:
        raise SystemExit(f"Unknown instance.type={short}; choose one of {list(INSTANCE_MAPPER)}")
    instance_type = INSTANCE_MAPPER[short]
    region = cfg["aws"]["region"]
    if region not in QUEUE_SUFFIX or instance_type not in QUEUE_SUFFIX[region]:
        raise SystemExit(f"No queue mapping for region={region} instance={instance_type}")
    queue_name = f"fss-{cfg['queue']['name']}-{QUEUE_SUFFIX[region][instance_type]}"
    print(f"Queue: {queue_name}")

    secrets = load_secrets("secrets.env")
    wandb_key = os.environ.get("WANDB_API_KEY") or secrets.get("WANDB_API_KEY")
    if cfg["wandb"]["enabled"] and not wandb_key and not local:
        raise SystemExit("Set $WANDB_API_KEY, add it to scripts/sagemaker/secrets.env, or disable wandb.")

    if dry_run or skip_build:
        account = ecr_account(region, cfg["aws"]["profile"])
        image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{cfg['image']['repo_name']}:latest"
        if dry_run:
            image_uri = f"<dry-run>/{cfg['image']['repo_name']}:latest"
    else:
        image_uri = build_and_push_image(cfg)
    print(f"Image: {image_uri}")

    os.environ["AWS_DEFAULT_REGION"] = region
    if local:
        from sagemaker.local import LocalSession
        sm_session = LocalSession()
    else:
        sm_session = sagemaker.Session(
            boto_session=boto3.session.Session(
                region_name=region, profile_name=cfg["aws"]["profile"]
            )
        )

    # All trainer config flows through env vars consumed by sm_entrypoint.sh.
    env = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "PYTHONPATH": "/opt/ml/code/src:/opt/ml/code/packages/openpi-client/src",
        "NCCL_DEBUG": "INFO",
        "FI_EFA_FORK_SAFE": "1",
        # openpi trainer args.
        "TRAIN_CONFIG": cfg["training"]["config"],
        "EXP_NAME": cfg["training"]["exp_name"],
        "TRAIN_ARGS": cfg["training"].get("extra_args", ""),
        "RESUME": "1" if cfg["training"].get("resume") else "0",
        "OVERWRITE": "1" if cfg["training"].get("overwrite") else "0",
    }
    if cfg["wandb"]["enabled"]:
        env["WANDB_PROJECT"] = cfg["wandb"]["project"]
        env["WANDB_MODE"] = "online"
        if wandb_key:
            env["WANDB_API_KEY"] = wandb_key
    else:
        env["WANDB_MODE"] = "disabled"
    # Forward HF_TOKEN, etc. from secrets.env.
    for k, v in secrets.items():
        env.setdefault(k, v)

    job_name = make_job_name(cfg["job"]["name_prefix"], cfg["job"]["user"])
    max_run_seconds = int(cfg["job"]["max_run_days"]) * 24 * 60 * 60

    # Build the inputs dict — one TrainingInput per channel so we can pin
    # input_mode per channel (FastFile streams without download).
    inputs = {
        name: TrainingInput(s3_data=ch["s3_uri"], input_mode=ch["input_mode"])
        for name, ch in cfg["data"]["channels"].items()
    }

    # Resolve and log every S3 URI tied to this job up front, so when you come
    # back later to download checkpoints / pull logs you don't have to
    # reconstruct the path from the job name.
    #
    # Layout under output.s3_prefix (= s3://.../openpi/checkpoints/):
    #   {s3_prefix}/{job_name}/{config}/{exp_name}/{step}/   (mirrored from
    #                                                         /opt/ml/checkpoints/)
    #   {s3_prefix}/{job_name}/output.tar.gz                  (SageMaker artifact)
    s3_prefix = cfg["output"]["s3_prefix"].rstrip("/")
    checkpoint_s3_uri = f"{s3_prefix}/{job_name}/"
    output_s3_uri = s3_prefix
    config_name = cfg["training"]["config"]
    exp_name = cfg["training"]["exp_name"]
    wandb_state = f"online (project={cfg['wandb']['project']})" if cfg["wandb"]["enabled"] else "disabled"
    print()
    print("=== Resolved S3 URIs for this job ===")
    print(f"job_name:           {job_name}")
    print(f"wandb:              {wandb_state}")
    for ch_name, ti in inputs.items():
        print(f"input  {ch_name:10s}: {ti.config['DataSource']['S3DataSource']['S3Uri']}  ({ti.config['InputMode']})")
    print(f"output checkpoints: {checkpoint_s3_uri}{config_name}/{exp_name}/")
    print(f"output artifacts:   {output_s3_uri}/{job_name}/output.tar.gz")
    print(f"download cmd:       aws s3 sync \\")
    print(f"                      {checkpoint_s3_uri}{config_name}/{exp_name}/ \\")
    print(f"                      ./checkpoints/{config_name}/{exp_name}/ --profile {cfg['aws']['profile']}")
    print("=====================================")
    print()

    # Use the lower-level Estimator (not PyTorch) because we already bake the
    # entrypoint and source into the image — we don't want SageMaker to upload
    # a source_dir or override SAGEMAKER_PROGRAM.
    estimator = Estimator(
        image_uri=image_uri,
        sagemaker_session=sm_session,
        base_job_name=sanitize(f"{cfg['job']['name_prefix']}-{cfg['job']['user']}"),
        role=role,
        instance_count=cfg["instance"]["count"],
        instance_type="local_gpu" if local else instance_type,
        job_name=job_name,
        checkpoint_local_path=None if local else cfg["output"]["checkpoint_local_path"],
        checkpoint_s3_uri=None if local else checkpoint_s3_uri,
        output_path=cfg["output"]["s3_prefix"],
        # torch_distributed launches one process per GPU; sm_entrypoint.sh
        # picks SM_NUM_GPUS / SM_HOST_COUNT off the env and runs torchrun.
        distribution={"torch_distributed": {"enabled": True}},
        max_run=max_run_seconds,
        environment=env,
        keep_alive_period_in_seconds=5 * 60,
        tags=[{"Key": t["key"], "Value": t["value"]} for t in cfg["job"]["tags"]],
        volume_size=cfg["job"]["volume_size"],
    )

    if dry_run:
        print(f"[dry-run] would submit job_name={job_name} to {queue_name}")
        for ch, ti in inputs.items():
            print(f"[dry-run]   channel {ch}: {ti.config['DataSource']['S3DataSource']['S3Uri']} ({ti.config['InputMode']})")
        print(f"[dry-run] env keys: {sorted(env)}")
        return

    if local:
        estimator.fit(inputs=inputs)
        return

    queue = TrainingQueue(queue_name=queue_name)
    queue.map(
        estimator,
        inputs=[inputs],
        job_names=[job_name],
        priority=cfg["queue"]["priority"],
        share_identifier=cfg["queue"]["share_identifier"],
        timeout={"attemptDurationSeconds": max_run_seconds},
    )
    print(f"Queued {job_name} -> {queue_name}")


if __name__ == "__main__":
    main()
