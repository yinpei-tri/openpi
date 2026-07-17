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
import json
import logging
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

# Entrypoints whose S3 checkpoint sync is RANK-AWARE (only the primary host uploads, with
# --delete), so they are safe to run with instance.count > 1. Non-listed JAX/PyTorch
# entrypoints sync from every host and could let a non-primary host wipe rank-0's files.
MULTINODE_SAFE_ENTRYPOINTS = {"sm_entrypoint_robocasa_multinode_jax.sh"}
# Entrypoints that honor --resume. The multi-node RoboCasa entrypoint deliberately does NOT
# (the trainer + loader reject a multi-node resume), so a resume request there would silently
# start fresh — the launcher rejects that combination instead.
RESUME_CAPABLE_ENTRYPOINTS = {"sm_entrypoint.sh", "sm_entrypoint_jax.sh"}


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


def _has_buildx() -> bool:
    """True if `docker buildx` is available (needed for the Dockerfile's BuildKit syntax
    — COPY --from / RUN --mount). The multi-node builds use it; single-node builds work
    either way (docker build on daemons with integrated BuildKit also works)."""
    return subprocess.run("docker buildx version", shell=True, capture_output=True).returncode == 0


def _s3_prefix_has_objects(s3_uri: str, aws_cfg: dict) -> bool:
    """True if any object exists under s3_uri. Read-only (list, 1 key). Used to guard
    against clobbering an existing experiment's checkpoints (the S3 dir is now keyed by the
    timestamp-free semantic exp_name). Returns False on any error (never blocks submit on a
    transient list failure — the guard is a courtesy, not a correctness gate)."""
    from urllib.parse import urlparse

    try:
        u = urlparse(s3_uri)
        session = boto3.Session(profile_name=aws_cfg.get("profile"), region_name=aws_cfg.get("region"))
        resp = session.client("s3").list_objects_v2(Bucket=u.netloc, Prefix=u.path.lstrip("/"), MaxKeys=1)
        return resp.get("KeyCount", 0) > 0
    except Exception as e:
        logging.warning(f"Could not check S3 for existing checkpoints ({s3_uri}): {e!r}; skipping collision guard.")
        return False


def ecr_account(region: str, profile: str) -> str:
    out = subprocess.check_output(
        [
            "aws",
            "--region",
            region,
            "--profile",
            profile,
            "sts",
            "get-caller-identity",
            "--query",
            "Account",
            "--output",
            "text",
        ],
        text=True,
    ).strip()
    if not out.isdigit():
        raise RuntimeError(f"Invalid account: {out!r}")
    return out


def image_tag(cfg: dict) -> str:
    """Immutable per-build tag. Defaults to a timestamp so concurrent builds from
    different machines/experiments never clobber each other's image (ECR `:latest` is
    a MUTABLE tag: whoever pushes last wins, and a QUEUED SageMaker job pulls whatever
    `:latest` resolves to at instance-boot — so another project pushing `:latest`
    between submit and boot silently swaps the container out from under this job).
    Override with image.tag in the config to pin a specific build."""
    return str(cfg["image"].get("tag") or datetime.now().strftime("%Y%m%d-%H%M%S"))


def build_and_push_image(cfg: dict, tag: str) -> str:
    region = cfg["aws"]["region"]
    profile = cfg["aws"]["profile"]
    repo = cfg["image"]["repo_name"]

    account = ecr_account(region, profile)
    # Push BOTH the immutable per-build tag (what the estimator pins to) and :latest
    # (convenience for --skip-build / manual pulls). The estimator uses the unique tag,
    # so a later :latest overwrite by another job cannot affect this run.
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}"
    latest = f"{account}.dkr.ecr.{region}.amazonaws.com/{repo}:latest"
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

    # Trainer entrypoint baked into the image: sm_entrypoint.sh (PyTorch, default)
    # or sm_entrypoint_jax.sh (JAX scripts/train.py). Set image.sm_entrypoint in config.
    sm_entrypoint = cfg["image"].get("sm_entrypoint", "sm_entrypoint.sh")

    run(login_dlc)
    # Prefer buildx (BuildKit) — the Dockerfile may use `COPY --from` / `RUN --mount`.
    # `--load` imports the result into the local daemon so the subsequent tag/push works
    # (the docker-container builder doesn't auto-load). Falls back to plain `docker build`
    # when buildx isn't present (daemons with integrated BuildKit handle the syntax too).
    build_cmd = "docker buildx build" if _has_buildx() else "docker build"
    load_flag = " --load" if _has_buildx() else ""
    run(
        f"{build_cmd} --progress=plain{load_flag} -f {dockerfile} "
        f"--build-arg AWS_REGION={region} --build-arg SM_ENTRYPOINT={sm_entrypoint} -t {repo} ."
    )
    run(f"docker tag {repo} {fullname}")
    run(f"docker tag {repo} {latest}")
    run(login_self)
    # Create the ECR repo if it doesn't exist (idempotent).
    run(
        f"aws --region {region} --profile {profile} ecr describe-repositories "
        f"--repository-names {repo} --no-cli-pager > /dev/null 2>&1 || "
        f"aws --region {region} --profile {profile} ecr create-repository "
        f"--repository-name {repo} --no-cli-pager"
    )
    run(f"docker push {fullname}")
    run(f"docker push {latest}")
    time.sleep(3)
    return fullname


# -----------------------------------------------------------------------------
# Job-name helpers
# -----------------------------------------------------------------------------
def sanitize(name: str) -> str:
    name = name.replace("_", "-").replace(".", "-")
    name = "".join(c if c.isalnum() or c == "-" else "" for c in name)
    return name.strip("-") or "job"


def make_job_name(prefix: str, user: str, exp_name: str = "") -> str:
    """SageMaker job name — kept semantic so it's readable in CloudWatch.

    Format: ``{user}-{prefix}-{exp_name}-{MM-DD-HH-MM-SS}`` (year dropped to save chars;
    MM-DD-HH-MM-SS is unique enough within a project). Injecting ``exp_name`` (e.g.
    "full400k") is what makes one run distinguishable from another at a glance — set a
    short, descriptive ``name_prefix`` (e.g. "pi05-robocasa") and a short ``exp_name``.
    """
    parts = [p for p in (user, prefix, exp_name) if p]
    base = sanitize("-".join(parts))
    stamp = datetime.now().strftime("%m-%d-%H-%M-%S")
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
# Per-job manifest (submitted_jobs/) — the durable record of every submission
# -----------------------------------------------------------------------------
def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _extract_arg(arg_str: str, flag: str) -> str | None:
    """Pull a `--flag=value` or `--flag value` out of a trainer extra_args string."""
    toks = arg_str.split()
    for i, t in enumerate(toks):
        if t == flag and i + 1 < len(toks):
            return toks[i + 1]
        if t.startswith(flag + "="):
            return t.split("=", 1)[1]
    return None


def _sync_norm_stats_from_channel(cfg: dict, *, dry_run: bool) -> None:
    """Install norm_stats.json from the robocasa data channel's S3 root into the baked
    assets dir, so the image ALWAYS ships the stats matched to the data this job trains
    on. Eliminates the manual, error-prone per-run swap. No-op if the channel/stats
    aren't found (leaves whatever is on disk). RoboCasa (system1) only.

    Layout: <channel s3 root>/norm_stats.json  ->  assets/<config>/robocasa_system1/norm_stats.json
    (the config's DataConfig asset_id is 'robocasa_system1').
    """
    ch = cfg.get("data", {}).get("channels", {}).get("robocasa")
    if not ch or "robocasa" not in cfg["training"]["config"]:
        return
    src = ch["s3_uri"].rstrip("/") + "/norm_stats.json"
    dst = REPO_ROOT / "assets" / cfg["training"]["config"] / "robocasa_system1" / "norm_stats.json"
    if dry_run:
        print(f"[norm_stats] would install {src} -> {dst.relative_to(REPO_ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    region = cfg["aws"]["region"]
    profile = cfg["aws"]["profile"]
    try:
        run(f"aws s3 cp {src} {dst} --region {region} --profile {profile} --only-show-errors")
        print(f"[norm_stats] installed {src} -> {dst.relative_to(REPO_ROOT)}")
    except Exception as e:
        raise SystemExit(
            f"[norm_stats] FAILED to fetch {src} (needed to bake matched norm_stats): {e}. "
            "Ensure the dataset root has norm_stats.json, or install it manually."
        ) from e


def _norm_stats_fingerprint(config_name: str) -> dict:
    """Summarize the norm_stats.json that WILL be baked (already installed on disk by
    _sync_norm_stats_from_channel): sha256 + per-key dims + a couple of summary values.
    Lets you verify at a glance which normalization a job used, and detect drift."""
    import hashlib

    path = REPO_ROOT / "assets" / config_name / "robocasa_system1" / "norm_stats.json"
    if not path.exists():
        return {"present": False}
    raw = path.read_bytes()
    fp = {"present": True, "sha256": hashlib.sha256(raw).hexdigest()[:16], "bytes": len(raw)}
    try:
        ns = json.loads(raw).get("norm_stats", {})
        for k, v in ns.items():
            m = v.get("mean") or []
            fp[k] = {"dim": len(m), "mean0": round(m[0], 5) if m else None}
    except Exception:
        pass
    return fp


def _write_job_manifest(
    *,
    job_name,
    semantic,
    queue_name,
    image_uri,
    env,
    inputs,
    cfg,
    config_name,
    exp_name,
    checkpoint_s3_uri,
    output_s3_uri,
    max_run_seconds,
) -> None:
    """Write a self-contained record of this submission to
    scripts/sagemaker/submitted_jobs/<semantic>/ (job.json for tooling, job.md for reading,
    norm_stats.json for reproducibility). ONE folder per experiment (keyed by the semantic
    exp_name, matching the S3 layout) instead of a flat pile of {job_name}.* files — so the
    dir stays navigable as runs accumulate, and the job-monitor skill can find a run by its
    semantic name. Captures everything you'd otherwise reconstruct later: job id, wandb
    project, data channels + URIs, image tag, config + key trainer args, checkpoint S3
    location, queue/priority, and the git commit."""
    out_dir = REPO_ROOT / "scripts" / "sagemaker" / "submitted_jobs" / semantic
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = {
        name: {"s3_uri": ti.config["DataSource"]["S3DataSource"]["S3Uri"], "input_mode": ti.config["InputMode"]}
        for name, ti in inputs.items()
    }
    ckpt_dir = f"{checkpoint_s3_uri}{config_name}/{exp_name}/"
    rec = {
        "job_name": job_name,
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "queue": queue_name,
        "priority": cfg["queue"]["priority"],
        "instance": {
            "type": cfg["instance"]["type"],
            "count": cfg["instance"]["count"],
            "volume_size_gb": cfg["job"]["volume_size"],
        },
        "max_run_days": cfg["job"]["max_run_days"],
        "image_uri": image_uri,
        "train_config": config_name,
        "exp_name": exp_name,
        "comment": cfg["training"].get("comment", ""),
        "train_args": cfg["training"].get("extra_args", ""),
        # How the container reads training data:
        #   "download-ebs" : entrypoint aws-s3-syncs the dataset to local EBS, reads local disk
        #                    (set via data.download_s3_uri; robust, no FUSE, fastest reads)
        #   "fastfile"     : reads the FastFile FUSE-mounted channel (streamed, no download)
        "data_loading": "download-ebs" if env.get("ROBOCASA_DATA_S3_URI") else "fastfile",
        "download_to_local": env.get("ROBOCASA_DATA_S3_URI", ""),
        # The JAX trainer logs to config.project_name, set by --project-name in extra_args
        # (NOT the yaml wandb.project, which only feeds the unused WANDB_PROJECT env). Report
        # the EFFECTIVE project the run actually appears under; fall back to the yaml value.
        "wandb_project": (
            _extract_arg(cfg["training"].get("extra_args", ""), "--project-name")
            or (cfg["wandb"]["project"] if cfg["wandb"]["enabled"] else "(disabled)")
        ),
        # The wandb RUN id is generated inside the container at init and written to
        # <ckpt_dir>/wandb_id.txt — recorded here as a pointer (not known at submit time).
        "wandb_id_file": f"{ckpt_dir}wandb_id.txt",
        "data_channels": channels,
        # Fingerprint (sha256 + dims + mean[0]) of the norm_stats baked into THIS job's
        # image, so you can verify/trace exactly which normalization was used.
        "norm_stats": _norm_stats_fingerprint(config_name),
        "semantic": semantic,
        "checkpoint_s3_dir": ckpt_dir,
        "output_artifacts": f"{output_s3_uri}/{job_name}/output.tar.gz",
        "ckpt_download_cmd": f"aws s3 sync {ckpt_dir} ./checkpoints/{config_name}/{exp_name}/ --profile {cfg['aws']['profile']}",
    }
    (out_dir / "job.json").write_text(json.dumps(rec, indent=2) + "\n")

    # Also preserve the EXACT norm_stats.json bytes baked into this job (not just the
    # fingerprint), so the run is fully reproducible from the record alone.
    ns_src = REPO_ROOT / "assets" / config_name / "robocasa_system1" / "norm_stats.json"
    if ns_src.exists():
        (out_dir / "norm_stats.json").write_text(ns_src.read_text())

    md = [
        f"# {job_name}",
        "",
        *([f"> {rec['comment']}", ""] if rec["comment"] else []),
        f"- **submitted:** {rec['submitted_at']}  (git `{rec['git_commit']}`)",
        f"- **queue:** {queue_name}  (priority {rec['priority']})",
        f"- **instance:** {rec['instance']['count']}x {rec['instance']['type']}, EBS {rec['instance']['volume_size_gb']}GB, max {rec['max_run_days']}d",
        f"- **image:** `{image_uri}`",
        f"- **config:** {config_name}  |  **exp_name:** {exp_name}",
        f"- **train_args:** `{rec['train_args']}`",
        f"- **data_loading:** {rec['data_loading']}"
        + (f" (from {rec['download_to_local']})" if rec["download_to_local"] else " (FastFile FUSE mount)"),
        f"- **wandb:** project `{rec['wandb_project']}`  (run id -> `{rec['wandb_id_file']}`)",
        "- **data channels:**",
        *[f"    - {n}: {c['s3_uri']} ({c['input_mode']})" for n, c in channels.items()],
        f"- **norm_stats:** sha256 `{rec['norm_stats'].get('sha256', 'n/a')}` "
        f"(state dim {rec['norm_stats'].get('state', {}).get('dim', '?')}, "
        f"actions dim {rec['norm_stats'].get('actions', {}).get('dim', '?')}; "
        f"full copy: `norm_stats.json`)",
        f"- **checkpoints:** {ckpt_dir}",
        f"- **download ckpts:** `{rec['ckpt_download_cmd']}`",
        "",
    ]
    (out_dir / "job.md").write_text("\n".join(md))
    print(f"Job record written: scripts/sagemaker/submitted_jobs/{semantic}/{{job.json,job.md,norm_stats.json}}")


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
    # The bare queue is `fss-{name}-{suffix}`, but when submission goes through the
    # cross-account `shared-sagemaker` scheduler the SubmitServiceJob jobQueue must be
    # the QUALIFIED name `shared-sagemaker__{region}__fss-...`. Allow an explicit
    # override (queue.full_name) for that case; else build the bare name.
    if cfg["queue"].get("full_name"):
        queue_name = cfg["queue"]["full_name"]
    else:
        queue_name = f"fss-{cfg['queue']['name']}-{QUEUE_SUFFIX[region][instance_type]}"
    print(f"Queue: {queue_name}")

    # SPOT vs RESERVED/on-demand queues need DIFFERENT env (see the env dict below):
    # the reserved queues (e.g. fss-vla-*) require SM_USE_RESERVED_CAPACITY=1 to admit the
    # job, but on a SPOT queue (fss-*-spot-*) that same flag makes the scheduler STOP the
    # job (it's asking for reserved capacity on a spot pool). So gate the flag on the queue
    # type. Detect spot by the "-spot-" segment in the resolved queue name.
    is_spot_queue = "-spot-" in queue_name
    print(f"Queue type: {'SPOT' if is_spot_queue else 'reserved/on-demand'}")

    # --- entrypoint / topology / resume safety gate --------------------------------------
    entrypoint_name = cfg["image"].get("sm_entrypoint", "sm_entrypoint.sh")
    count = int(cfg["instance"]["count"])
    if count > 1 and entrypoint_name not in MULTINODE_SAFE_ENTRYPOINTS:
        raise SystemExit(
            f"instance.count={count} but entrypoint '{entrypoint_name}' is NOT multi-node safe "
            f"(it syncs checkpoints from every host with --delete; a non-primary host could wipe "
            f"rank-0's checkpoint). Use a rank-aware entrypoint {sorted(MULTINODE_SAFE_ENTRYPOINTS)} "
            f"(e.g. config_robocasa_multinode.yaml), or set instance.count=1."
        )
    if cfg["training"].get("resume") and entrypoint_name not in RESUME_CAPABLE_ENTRYPOINTS:
        raise SystemExit(
            f"training.resume=true but entrypoint '{entrypoint_name}' does not support resume "
            f"(it would silently start a FRESH run). Resume is single-node only; for multi-node, "
            f"start fresh (EMA-only, no train_state)."
        )

    secrets = load_secrets("secrets.env")
    wandb_key = os.environ.get("WANDB_API_KEY") or secrets.get("WANDB_API_KEY")
    if cfg["wandb"]["enabled"] and not wandb_key and not local:
        raise SystemExit("Set $WANDB_API_KEY, add it to scripts/sagemaker/secrets.env, or disable wandb.")

    # Auto-install norm_stats from the dataset's S3 root into the baked assets so it is
    # ALWAYS matched to the robocasa data channel — no manual per-run edit, no "wrong
    # stats baked" footgun. Runs only when we actually build (skip-build reuses whatever
    # is already in the image). Dry-run just reports what it would fetch.
    _sync_norm_stats_from_channel(cfg, dry_run=dry_run or skip_build)

    # Immutable per-build tag so a concurrent :latest push (e.g. a LIBERO experiment on
    # another machine) can't swap this job's container at instance-boot. --skip-build
    # reuses :latest (must set image.tag to pin a specific prior build instead).
    tag = image_tag(cfg)
    if dry_run or skip_build:
        account = ecr_account(region, cfg["aws"]["profile"])
        pin = cfg["image"].get("tag") or "latest"
        image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{cfg['image']['repo_name']}:{pin}"
        if dry_run:
            image_uri = f"<dry-run>/{cfg['image']['repo_name']}:{pin}"
    else:
        image_uri = build_and_push_image(cfg, tag)
    print(f"Image: {image_uri}")

    os.environ["AWS_DEFAULT_REGION"] = region
    if local:
        from sagemaker.local import LocalSession

        sm_session = LocalSession()
    else:
        sm_session = sagemaker.Session(
            boto_session=boto3.session.Session(region_name=region, profile_name=cfg["aws"]["profile"])
        )

    # All trainer config flows through env vars consumed by sm_entrypoint.sh.
    env = {
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
    # SM_USE_RESERVED_CAPACITY: required on RESERVED/on-demand queues (fss-vla-*) to admit
    # the job; but on a SPOT queue it causes the scheduler to STOP the job (reserved-capacity
    # request against a spot pool). So set it ONLY for non-spot queues, and leave it UNSET on
    # spot. (This is why an earlier 2-node spot submit never scheduled.)
    if not is_spot_queue:
        env["SM_USE_RESERVED_CAPACITY"] = "1"
    # Download-to-local data mode (JAX RoboCasa): if data.download_s3_uri is set, the
    # entrypoint `aws s3 sync`s that s3:// dataset root to local EBS once and reads from
    # local disk (no FastFile FUSE mount -> no ENOTCONN mid-run, fastest reads). Requires
    # volume_size to fit the dataset + checkpoints. Unset -> read from the FastFile mount.
    if cfg["data"].get("download_s3_uri"):
        env["ROBOCASA_DATA_S3_URI"] = cfg["data"]["download_s3_uri"]

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

    job_name = make_job_name(cfg["job"]["name_prefix"], cfg["job"]["user"], cfg["training"].get("exp_name", ""))
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
    #   {s3_prefix}/{semantic}/{config}/{exp_name}/{step}/   (the checkpoints — mirrored
    #                                                         from /opt/ml/checkpoints/)
    #   {s3_prefix}/_artifacts/{job_name}/output.tar.gz       (SageMaker debug/profiler
    #                                                         tarballs — kept OUT of the
    #                                                         semantic dir so the entrypoint's
    #                                                         `--delete` checkpoint sync can
    #                                                         never touch them, and they don't
    #                                                         clutter the checkpoints root)
    # `semantic` = sanitized exp_name (e.g. "subset0713-30k-progact") — ONE stable, readable
    # dir per experiment (vs the old timestamped job_name, which scattered every run + every
    # AWSBatch* artifact across the flat root). The CloudWatch job_name keeps its timestamp
    # (uniqueness for log-stream matching); the S3 dir drops it for readability, so we guard
    # against clobbering an existing experiment's checkpoints below.
    config_name = cfg["training"]["config"]
    exp_name = cfg["training"]["exp_name"]
    s3_prefix = cfg["output"]["s3_prefix"].rstrip("/")
    semantic = sanitize(exp_name)
    checkpoint_s3_uri = f"{s3_prefix}/{semantic}/"
    # SageMaker's own output_path (debug-output/, profiler-output/, output.tar.gz) — parked
    # under _artifacts/<job_name>/ so timestamped runs stay distinct there without polluting
    # the checkpoints root.
    artifacts_s3_uri = f"{s3_prefix}/_artifacts"
    output_s3_uri = artifacts_s3_uri

    # Two independent properties of the entrypoint:
    #   is_jax            -> JAX trainer: ONE process per node (single- OR multi-node), NO
    #                        torch_distributed (the trainer forms the mesh itself via
    #                        jax.distributed.initialize).
    #   self_manages_ckpt -> the entrypoint runs its OWN `aws s3 sync` from local EBS, so
    #                        SageMaker-managed /opt/ml/checkpoints sync must be OFF (that
    #                        sidecar races orbax's write-then-read-back on the eventually-
    #                        consistent mount and crashes finalize; for multi-node it is a
    #                        per-node dir, and two nodes syncing it corrupts the checkpoint).
    # ALL JAX entrypoints (single-node sm_entrypoint_jax.sh AND the multi-node
    # *_multinode_jax.sh) self-manage the sync + skip torch_distributed, so gate on the
    # "_jax.sh" SUFFIX rather than the exact single-node filename. PyTorch keeps managed
    # checkpointing + torch_distributed.
    _entrypoint = cfg["image"].get("sm_entrypoint", "sm_entrypoint.sh")
    is_jax = _entrypoint.endswith("_jax.sh")
    self_manages_ckpt = is_jax
    env["CHECKPOINT_S3_URI"] = checkpoint_s3_uri
    # Periodic background-sync interval for the JAX entrypoint (seconds). Optional in
    # the config; the entrypoint defaults to 1800 if unset.
    if cfg["output"].get("ckpt_sync_interval") is not None:
        env["CKPT_SYNC_INTERVAL"] = str(cfg["output"]["ckpt_sync_interval"])
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

    # Collision guard: the S3 checkpoint dir is now keyed by the (timestamp-free) semantic
    # exp_name, so a second run with the SAME exp_name would write into the same
    # {semantic}/{config}/{exp}/ tree and the entrypoint's `--delete` final sync could
    # clobber the earlier run's checkpoints. Refuse unless --overwrite. Skipped on dry-run /
    # local (no real submit) and when resuming (resume intentionally reuses the dir). Never
    # deletes anything — just checks + aborts. Requires listing S3, so guard on real submit.
    if not (dry_run or local) and not cfg["training"].get("overwrite") and not cfg["training"].get("resume"):
        existing = _s3_prefix_has_objects(f"{checkpoint_s3_uri}{config_name}/{exp_name}/", cfg["aws"])
        if existing:
            raise SystemExit(
                f"Refusing to submit: S3 checkpoint dir already has objects at\n"
                f"  {checkpoint_s3_uri}{config_name}/{exp_name}/\n"
                f"exp_name '{exp_name}' is not unique (the S3 path drops the timestamp). Pick a new "
                f"exp_name, or set training.overwrite=true to reuse/replace it."
            )

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
        # Managed checkpoint sync OFF for entrypoints that self-manage it (all JAX paths,
        # single- AND multi-node — see self_manages_ckpt above). PyTorch keeps it ON.
        checkpoint_local_path=None if (local or self_manages_ckpt) else cfg["output"]["checkpoint_local_path"],
        checkpoint_s3_uri=None if (local or self_manages_ckpt) else checkpoint_s3_uri,
        # SageMaker's own output artifacts (output.tar.gz, debug-output/, profiler-output/)
        # land under {s3_prefix}/_artifacts/{job_name}/ — OFF the checkpoints root so they
        # don't clutter it with AWSBatch*/ dirs and can't collide with the semantic ckpt dirs.
        output_path=output_s3_uri,
        # torch_distributed launches one process per GPU (PyTorch runs torchrun). JAX
        # trainers run ONE process per NODE (each owns all local GPUs) and form the mesh
        # via jax.distributed.initialize, so they must NOT use torch_distributed — even
        # multi-node. Gate on the "_jax" entrypoint suffix (is_jax).
        distribution=({} if is_jax else {"torch_distributed": {"enabled": True}}),
        max_run=max_run_seconds,
        environment=env,
        keep_alive_period_in_seconds=5 * 60,
        tags=[{"Key": t["key"], "Value": t["value"]} for t in cfg["job"]["tags"]],
        volume_size=cfg["job"]["volume_size"],
    )

    if dry_run:
        print(f"[dry-run] would submit job_name={job_name} to {queue_name}")
        for ch, ti in inputs.items():
            print(
                f"[dry-run]   channel {ch}: {ti.config['DataSource']['S3DataSource']['S3Uri']} ({ti.config['InputMode']})"
            )
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

    # Persist a per-job record so we never lose track of which job used which data /
    # image / config / where its checkpoints + wandb live. One file per job under
    # scripts/sagemaker/submitted_jobs/ — no in-place file mutation to reconstruct later.
    _write_job_manifest(
        job_name=job_name,
        semantic=semantic,
        queue_name=queue_name,
        image_uri=image_uri,
        env=env,
        inputs=inputs,
        cfg=cfg,
        config_name=config_name,
        exp_name=exp_name,
        checkpoint_s3_uri=checkpoint_s3_uri,
        output_s3_uri=output_s3_uri,
        max_run_seconds=max_run_seconds,
    )


if __name__ == "__main__":
    main()
