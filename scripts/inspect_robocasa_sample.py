"""Inspect RoboCasa System1 WebDataset samples — raw shard vs model-ready.

Shows the full data-processing pipeline for a few samples, through the SAME code
training uses, so what you see is what the model gets:

  RAW    : straight out of RoboCasaWebDataset._build_sample (pre-transform) — decoded
           images, raw+baked lean state, chosen lean action chunk, prompt fields,
           conditioning tags, progress labels.
  MODEL  : after data_transforms -> Normalize -> model_transforms — the assembled
           prompt STRING (decoded from tokens), normalized 28-d state, 11->32 padded
           action, tokenized prompt, progress_class. These are the model inputs.

Usage:
  uv run scripts/inspect_robocasa_sample.py                 # 3 samples, both stages
  uv run scripts/inspect_robocasa_sample.py -n 5
  uv run scripts/inspect_robocasa_sample.py --config pi05_robocasa_system1
  uv run scripts/inspect_robocasa_sample.py --dump-images /tmp/rc  # save the JPEGs
  # data location: --shards <dir> (else $ROBOCASA_SHARDS_DIR, else the config default)
"""

import argparse
import dataclasses
import os
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


def _fmt(arr) -> str:
    a = np.asarray(arr)
    if a.ndim == 0:
        return f"scalar={a}"
    flat = a.reshape(-1)
    head = np.array2string(flat[:10], precision=3, separator=", ")
    return f"shape={a.shape} dtype={a.dtype} [:10]={head}"


def _decode_prompt(out: dict, tok) -> str | None:
    if "tokenized_prompt" not in out or tok is None:
        return None
    ids = np.asarray(out["tokenized_prompt"]).tolist()
    mask = np.asarray(out.get("tokenized_prompt_mask", np.ones_like(out["tokenized_prompt"]))).astype(bool)
    real = [t for t, m in zip(ids, mask) if m]
    return tok.decode(real)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_robocasa_system1")
    ap.add_argument("--shards", default=None, help="override shards dir (else config/env default)")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-images", default=None, help="dir to save the sample JPEGs")
    args = ap.parse_args()

    import openpi.training.config as _config
    from openpi.training import robocasa_webdataset as wds
    from openpi import transforms as _t
    from openpi.transforms import compose
    from openpi.models import tokenizer as _tk

    tc = _config.get_config(args.config)
    dc = tc.data.create(tc.assets_dirs, tc.model)
    shards = args.shards or os.environ.get("ROBOCASA_SHARDS_DIR") or tc.data.shards

    # Full transform stack (exactly what the data loader applies).
    tfs = [
        *dc.repack_transforms.inputs,
        *dc.data_transforms.inputs,
        _t.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
        *dc.model_transforms.inputs,
    ]
    fn = compose(tfs)

    # Reuse the config's exact WebDatasetConfig; local path + tiny buffer for immediacy.
    cfg = dataclasses.replace(dc.robocasa_webdataset_settings, shards=shards,
                              seed=args.seed, shuffle_buffer=1, shuffle_initial=1)
    ds = wds.RoboCasaWebDataset(cfg)
    tok = _tk.PaligemmaTokenizer(max_len=tc.model.max_token_len)._tokenizer

    print(f"# config={args.config}  shards={shards}  len≈{len(ds)}")
    print(f"# prompt_source={cfg.prompt_source} pad={cfg.subgoal_action_pad} repad={cfg.repad_actions} "
          f"anchors={cfg.use_anchor_images} anchor_state={cfg.include_anchor_state} "
          f"progress_mode={tc.model.progress_mode}({tc.model.progress_num_classes})\n")

    for i, raw in zip(range(args.n), iter(ds)):
        print("=" * 78)
        print(f"SAMPLE {i}")
        print("=" * 78)

        # ---------- RAW (pre-transform) ----------
        print("--- RAW (out of RoboCasaWebDataset, pre-transform) ---")
        for k in ("prompt", "task_goal", "quality", "est_length", "executed_step", "gripper_flag",
                  "progress_frac", "progress_class", "subgoal_start", "subgoal_end", "frame_index"):
            if k in raw:
                v = raw[k]
                print(f"  {k:15s}: {v!r}" if not isinstance(v, np.ndarray) else f"  {k:15s}: {_fmt(v)}")
        for k, v in raw.items():
            if isinstance(v, np.ndarray) and k.startswith("observation/"):
                tag = "IMG" if v.ndim == 3 else "   "
                print(f"  {tag} {k:30s}: {_fmt(v)}")
        if "actions" in raw:
            act = np.asarray(raw["actions"])
            print(f"      actions (lean 11-d)          : {_fmt(act)}")
            print(f"        last real row : {np.round(act[act.any(-1)][-1], 3) if act.any() else act[-1]}")
            print(f"        final row     : {np.round(act[-1], 3)}  (pad rows hold ctrl_mode+gripper)")

        if args.dump_images:
            from PIL import Image
            d = pathlib.Path(args.dump_images); d.mkdir(parents=True, exist_ok=True)
            for k, v in raw.items():
                if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3:
                    nm = k.replace("observation/", "").replace("/", "_")
                    Image.fromarray(v.astype(np.uint8)).save(d / f"sample{i}_{nm}.jpg")
            print(f"  (images saved to {args.dump_images})")

        # ---------- MODEL (post full transform) ----------
        out = fn(dict(raw))
        print("\n--- MODEL INPUTS (post data->normalize->model transforms) ---")
        prompt = _decode_prompt(out, tok)
        if prompt is not None:
            n_tok = int(np.asarray(out["tokenized_prompt_mask"]).astype(bool).sum())
            print(f"  PROMPT STRING ({n_tok} tokens / cap {tc.model.max_token_len}):")
            print("    " + prompt.replace("\n", "\n    "))
        for k in ("state", "actions", "tokenized_prompt", "tokenized_prompt_mask",
                  "progress_frac", "progress_class"):
            if k in out and out[k] is not None:
                print(f"  {k:22s}: {_fmt(out[k])}")
        imgs = out.get("image", {})
        if isinstance(imgs, dict):
            print(f"  image keys           : {list(imgs)}")
        print()


if __name__ == "__main__":
    main()
