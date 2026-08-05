#!/usr/bin/env python
"""Extract compact, DURABLE episode-eval results per checkpoint into
eval_results/episode_results/<method>.json — so the eval results survive deleting the big
per-episode video/steps dirs under eval_results/episode/<method>/.

Each method's episode/<method>/index.json is already the compact summary (~200 KB: per-episode
episode_success / task_name / seconds + totals) vs. ~9 GB of videos. This copies that index into
episode_results/, and if a method's index is MISSING or PARTIAL (fewer episodes than on disk),
rebuilds it from the per-episode episode.json first.

/stats reads episode_results/ first (durable), falling back to episode/<m>/index.json. Re-run this
any time after an eval finishes (idempotent). Run:
    python scripts/extract_episode_results.py
"""
import json
import glob
import shutil
from pathlib import Path

EP_ROOT = Path("eval_results/episode")
OUT = Path("eval_results/episode_results")


def _rebuild_from_disk(mdir: Path) -> dict:
    eps = []
    for f in sorted(glob.glob(str(mdir / "*" / "episode.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        # episode.json persists its own `seconds` (newer runs), so a rebuild keeps per-episode
        # timing even when the parallel-shard index.json race lost it. Older docs -> None.
        eps.append(dict(episode_id=d.get("episode_id"), task_name=d.get("task_name"),
                        episode_success=d.get("episode_success", d.get("sim_success_final")),
                        n_advanced=d.get("n_advanced"), n_subgoals=d.get("n_subgoals"),
                        seconds=d.get("seconds")))
    done = [e["seconds"] for e in eps if e.get("seconds") is not None]
    total_s = round(sum(done), 1) if done else None
    return dict(eval_kind="episode", method=mdir.name, n_episodes=len(eps),
                total_seconds=total_s,
                avg_seconds_per_episode=round(total_s / len(done), 1) if done else None,
                episodes=sorted(eps, key=lambda e: e.get("episode_id") or ""))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not EP_ROOT.is_dir():
        print(f"no {EP_ROOT}")
        return
    for mdir in sorted(EP_ROOT.iterdir()):
        if not mdir.is_dir():
            continue
        idx_f = mdir / "index.json"
        idx = None
        if idx_f.is_file():
            try:
                idx = json.loads(idx_f.read_text())
            except Exception:
                idx = None
        n_disk = len(glob.glob(str(mdir / "*" / "episode.json")))
        n_idx = len(idx.get("episodes", [])) if idx else 0
        # Prefer the index (durable summary + has seconds). Rebuild only if index is missing OR
        # strictly SMALLER than what's on disk (partial — e.g. shard race). If the index is >= disk
        # (e.g. videos already cleaned but index retained), keep the index — it's the surviving record.
        if idx is not None and n_idx >= n_disk:
            out_doc = idx
            src = f"index ({n_idx} eps)"
        else:
            out_doc = _rebuild_from_disk(mdir)
            src = f"rebuilt from disk ({out_doc['n_episodes']} eps; index had {n_idx})"
        n_succ = sum(1 for e in out_doc.get("episodes", []) if e.get("episode_success"))
        (OUT / f"{mdir.name}.json").write_text(json.dumps(out_doc, indent=1))
        print(f"{mdir.name}: {src} -> episode_results/{mdir.name}.json  ({n_succ} success)")


if __name__ == "__main__":
    main()
