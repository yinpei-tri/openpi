"""Trace ONE RoboCasa sample through the pipeline, one transform at a time.

Loads the FIRST sample of the FIRST shard (deterministic; no shuffle) and walks it
through every stage so you can see exactly how a raw shard frame becomes model input:

  STAGE 0  raw tar members (what's on disk for this key)
  STAGE 1  RoboCasaWebDataset._build_sample  (decode + pick chunk/prompt/labels)
  STAGE 2  RobocasaInputs                     (lean state recompute+verify, prompt build)
  STAGE 3  Normalize                          (quantile-normalize state + actions)
  STAGE 4  model_transforms                   (resize imgs, tokenize prompt, pad action)

Usage:
  uv run scripts/trace_robocasa_sample.py                    # first sample, first shard
  uv run scripts/trace_robocasa_sample.py --sample-index 5   # 6th sample in the shard
  uv run scripts/trace_robocasa_sample.py --shard 3          # 4th shard
"""

import argparse
import dataclasses
import glob
import io
import json
import os
import pathlib
import tarfile

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=140)


def _hdr(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _show(name, v, *, full=False):
    a = np.asarray(v)
    if a.dtype.kind in ("U", "S", "O"):
        print(f"  {name:34s} = {v!r}")
    elif a.ndim == 0:
        print(f"  {name:34s} = {a}")
    elif a.ndim == 3:  # image
        print(f"  {name:34s} : image {a.shape} {a.dtype}  mean={a.mean():.1f}")
    else:
        s = np.array2string(a if full else a.reshape(-1)[:16])
        print(f"  {name:34s} : {a.shape} {a.dtype}\n      {s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_robocasa_system1")
    ap.add_argument("--shards", default=None)
    ap.add_argument("--shard", type=int, default=0, help="which shard (sorted index)")
    ap.add_argument("--sample-index", type=int, default=0, help="which sample within the shard")
    args = ap.parse_args()

    import openpi.training.config as _config
    from openpi.training import robocasa_webdataset as wds
    from openpi import transforms as _t
    from openpi.models import tokenizer as _tk

    tc = _config.get_config(args.config)
    dc = tc.data.create(tc.assets_dirs, tc.model)
    shards_dir = args.shards or os.environ.get("ROBOCASA_SHARDS_DIR") or tc.data.shards
    all_shards = sorted(glob.glob(f"{shards_dir}/shard_*.tar"))
    if not all_shards:
        raise SystemExit(
            f"No shards found at {shards_dir!r}. Set --shards <dir> or export ROBOCASA_SHARDS_DIR."
        )
    if args.shard >= len(all_shards):
        raise SystemExit(f"--shard {args.shard} out of range: only {len(all_shards)} shards at {shards_dir!r}.")
    shard = all_shards[args.shard]

    # ---- read the raw tar members for the chosen sample (no shuffle, deterministic) ----
    members = None
    with tarfile.open(shard, "r|*") as tar:
        cur, grp, idx = None, {}, -1
        for m in tar:
            if not m.isfile():
                continue
            key, suf = m.name.split(".", 1)
            if cur is not None and key != cur:
                idx += 1
                if idx == args.sample_index:
                    members = grp
                    break
                grp = {}
            cur = key
            grp[suf] = tar.extractfile(m).read()
        if members is None and idx + 1 == args.sample_index:
            members = grp

    _hdr(f"STAGE 0 — raw tar members  (shard={pathlib.Path(shard).name}, sample #{args.sample_index})")
    for suf, b in members.items():
        print(f"  {suf:16s} {len(b):>10,d} bytes")
    meta = json.loads(members["meta.json"])
    print("\n  meta.json:")
    for k, v in meta.items():
        print(f"    {k:22s}: {v!r}")
    arrays = np.load(io.BytesIO(members["arrays.npz"]))
    print("\n  arrays.npz:")
    for k in arrays.files:
        _show("    " + k, arrays[k])

    # ---- STAGE 1: RoboCasaWebDataset._build_sample ----
    cfg = dataclasses.replace(dc.robocasa_webdataset_settings, shards=shards_dir)
    ds = wds.RoboCasaWebDataset(cfg)  # real constructor -> correct anchor-root setup
    import random as _random
    sample = ds._build_sample(members, _random.Random(0))

    _hdr("STAGE 1 — _build_sample (decode images, pick action chunk, prompt fields, labels)")
    for k, v in sample.items():
        _show(k, v)

    # ---- run transforms one at a time ----
    stages = [
        ("STAGE 2 — RobocasaInputs (lean recompute+verify, build prompt, drop torso)",
         dc.data_transforms.inputs),
        ("STAGE 3 — Normalize (quantile-normalize state + actions to ~[-1,1])",
         [_t.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm)]),
        ("STAGE 4 — model_transforms (resize 224, tokenize prompt+state, pad action->32)",
         dc.model_transforms.inputs),
    ]
    data = dict(sample)
    for title, tfs in stages:
        for tf in tfs:
            data = tf(data)
        _hdr(title)
        for k, v in data.items():
            if k == "image" and isinstance(v, dict):
                for ik, iv in v.items():
                    _show(f"image/{ik}", iv)
            elif k == "image_mask" and isinstance(v, dict):
                print(f"  image_mask                         = {{{', '.join(f'{k2}:{np.asarray(v2)}' for k2,v2 in v.items())}}}")
            else:
                _show(k, v)

    # ---- decode the tokenized prompt back to text ----
    tok = _tk.PaligemmaTokenizer(max_len=tc.model.max_token_len)._tokenizer
    if "tokenized_prompt" in data:
        ids = np.asarray(data["tokenized_prompt"]).tolist()
        mask = np.asarray(data["tokenized_prompt_mask"]).astype(bool)
        real = [t for t, m in zip(ids, mask) if m]
        _hdr(f"FINAL PROMPT STRING (decoded from {len(real)} tokens / cap {tc.model.max_token_len})")
        print(tok.decode(real))


if __name__ == "__main__":
    main()
