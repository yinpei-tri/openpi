"""Build a tiny LOCAL RoboCasa System1 fixture for debugging: download ONE S3 shard,
keep the first N samples (rewrite into a trimmed tar), and fetch ONLY the anchor images
those samples reference. Produces the `shards/` + `anchors/` sibling layout the loader
expects, so you can point ROBOCASA_SHARDS_DIR at it and iterate with zero S3 latency.

    AWS_PROFILE=sagemaker uv run python scripts/build_robocasa_debug_fixture.py \
        --n 100 --out debug_data

Result:
    debug_data/shards/shard_debug_000000.tar
    debug_data/shards/manifest.jsonl
    debug_data/anchors/<anchor_key>.<cam>.jpg ...
"""

import argparse
import io
import json
import pathlib
import tarfile
from urllib.parse import urlparse

import boto3

CAM_KEYS = ("scene_left", "scene_right", "wrist")
S3_ROOT = "s3://tri-ml-datasets-uw2/yinpeidai/robocasa_final_training_shard/system1_full_0711"
SHARD = "shards/shard_w00_000000.tar"


def s3_get(client, s3_uri: str) -> bytes:
    u = urlparse(s3_uri)
    return client.get_object(Bucket=u.netloc, Key=u.path.lstrip("/"))["Body"].read()


def iter_groups(tar_bytes: bytes):
    """Yield (key, {member_name: bytes}) grouped by the sample key, preserving order."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r|*") as tar:
        cur_key, group = None, {}
        for m in tar:
            if not m.isfile():
                continue
            key, _suffix = m.name.split(".", 1)
            if cur_key is not None and key != cur_key:
                yield cur_key, group
                group = {}
            cur_key = key
            group[m.name] = tar.extractfile(m).read()
        if group:
            yield cur_key, group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="debug_data")
    args = ap.parse_args()

    out = pathlib.Path(args.out).resolve()
    (out / "shards").mkdir(parents=True, exist_ok=True)
    (out / "anchors").mkdir(parents=True, exist_ok=True)

    client = boto3.client("s3")
    print(f"Downloading shard {SHARD} ...")
    tar_bytes = s3_get(client, f"{S3_ROOT}/{SHARD}")
    print(f"  got {len(tar_bytes)/1e6:.1f} MB")

    # 1) Keep the first N samples; rewrite into a trimmed tar (members contiguous).
    out_tar = out / "shards" / "shard_debug_000000.tar"
    anchor_keys: set[str] = set()
    kept = 0
    with tarfile.open(out_tar, mode="w") as w:
        for key, group in iter_groups(tar_bytes):
            if kept >= args.n:
                break
            # record anchor key from meta.json
            meta_name = next((n for n in group if n.endswith("meta.json")), None)
            if meta_name is None:
                continue
            meta = json.loads(group[meta_name])
            ak = meta.get("anchor_key")
            if ak:
                anchor_keys.add(ak)
            # write all members of this sample, preserving name/order
            for name, data in group.items():
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                w.addfile(info, io.BytesIO(data))
            kept += 1
    print(f"Wrote {kept} samples -> {out_tar}")

    # 2) manifest.jsonl so the loader's len() works.
    (out / "shards" / "manifest.jsonl").write_text(
        json.dumps({"shard": "shard_debug_000000.tar", "num_samples": kept}) + "\n"
    )

    # 3) Download only the anchor images those samples reference.
    print(f"Fetching anchors for {len(anchor_keys)} unique anchor keys ({len(anchor_keys)*3} images)...")
    got, miss = 0, 0
    for ak in sorted(anchor_keys):
        for cam in CAM_KEYS:
            rel = f"{ak}.{cam}.jpg"
            dst = out / "anchors" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                dst.write_bytes(s3_get(client, f"{S3_ROOT}/anchors/{rel}"))
                got += 1
            except Exception as e:  # noqa: BLE001
                miss += 1
                print(f"  MISS {rel}: {e}")
    print(f"Anchors: {got} downloaded, {miss} missing.")
    print(f"\nDone. Fixture at: {out}")
    print(f"  shards:  {out}/shards  ({kept} samples)")
    print(f"  anchors: {out}/anchors ({got} images)")
    print(f"\nUse with:  ROBOCASA_SHARDS_DIR={out}/shards")


if __name__ == "__main__":
    main()
