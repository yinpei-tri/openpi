"""System1+System2 evaluation from frozen procedural RoboCasa resets, without LeRobot data.

LeRobot is only one way to supply an initial MuJoCo model/state and ``ep_meta``. This entry point
creates those inputs directly from RoboCasa's task class using a fixed seed, then reuses
``combined_eval``'s policy loop unchanged. It is suitable for:

* the RoboCasa365 tasks that have an environment but no published simulated trajectory dataset;
* fresh-seed evaluation of existing target tasks, including the 16 composite-unseen tasks;
* paired model comparisons: use the same frozen reset bank for every model arm.

By default ``--episodes 0-29 --seed-base 100000`` means actual seeds 100000..100029. Target scene
and object-instance splits are used, matching the official target setup. Every result records the
actual selected layout/style and reset hashes. Generate reset bundles once, then use
``--load-reset-root`` for every model arm: this replays byte-identical model XML, initial simulator
state, and ep_meta even if RoboCasa's seed-generation behavior changes. The numeric seed is reset
provenance and a stable lookup key; the frozen bundle, not regeneration from that number, is the
exact benchmark input. No demonstration actions or videos are read.

Example:

    python examples/robocasa/seeded_combined_eval.py \
      --task AddToSoupPot --episodes 0-29 --seed-base 100000 \
      --task-limits-json examples/robocasa/procedural_unseen_task_limits.json \
      --s1-dir ... --s2-dir ... --norm-stats ...

Reset-only smoke test (does not contact either policy server):

    python examples/robocasa/seeded_combined_eval.py \
      --task AddToSoupPot --episodes 0 --reset-only

Fair paired evaluation uses a frozen reset bank:

    # Run once. Existing complete bundles are verified and reused, not replaced.
    python examples/robocasa/seeded_combined_eval.py \
      --task AddToSoupPot --episodes 0-29 --seed-base 100000 --reset-only \
      --reset-bundle-root /path/to/procedural_reset_bank

    # Use the same command for every model arm, changing only model/method arguments.
    python examples/robocasa/seeded_combined_eval.py \
      --task AddToSoupPot --episodes 0-29 --seed-base 100000 \
      --load-reset-root /path/to/procedural_reset_bank ...
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import random
import subprocess
import textwrap
import traceback

import numpy as np
from PIL import Image
from PIL import ImageDraw

import combined_eval as CE
from robocasa.scripts.eval.subtask_env import CAMERA_NAMES
from robocasa.utils.env_utils import create_env


DEFAULT_SEED_BASE = 100_000
RESET_SPLIT_PREFIX = "procedural"
_IMAGE_FILES = {
    "robot0_agentview_left": "initial_agentview_left.png",
    "robot0_agentview_right": "initial_agentview_right.png",
    "robot0_eye_in_hand": "initial_eye_in_hand.png",
}
_RESET_FILES = (
    "reset.json", "ep_meta.json", "task_goal.txt", "initial_state.npz", "model.xml.gz",
    *_IMAGE_FILES.values(), "initial_views.png",
)
_DATASET_RESET_TO = CE.reset_to


@contextmanager
def _isolated_reset_rng(seed: int):
    """Make legacy/global RNG calls deterministic without leaking state to the eval process.

    RoboCasa's environment-local ``Generator`` is seeded by ``create_env``. A few fixture and
    collision-geometry helpers still use either the legacy process-global ``np.random`` API or
    ``np.random.default_rng()`` without an argument, however. Seed and temporarily intercept those
    two fallback paths while the scene is constructed, then restore the caller's RNG state.

    Explicit ``default_rng(some_seed)`` calls retain their normal semantics. Only argument-free
    calls receive deterministic child seeds derived from this reset's seed.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    original_default_rng = np.random.default_rng
    fallback_rng = original_default_rng(np.random.SeedSequence([int(seed), 0x524F424F]))

    def deterministic_default_rng(seed_arg=None):
        if seed_arg is None:
            seed_arg = int(fallback_rng.integers(0, 2**32, dtype=np.uint32))
        return original_default_rng(seed_arg)

    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng = deterministic_default_rng
    try:
        yield
    finally:
        np.random.default_rng = original_default_rng
        np.random.set_state(np_state)
        random.setstate(py_state)


def _git_revision(repo: Path) -> str:
    return (subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip() or "unknown")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fake_episode_path(task: str) -> Path:
    """Shape only: combined_eval uses this to derive a task name, never reads it."""
    return Path("/procedural") / "composite" / task / "seeded" / "lerobot"


def _set_base_argument(parser: argparse.ArgumentParser, dest: str, **updates) -> None:
    action = next(a for a in parser._actions if a.dest == dest)
    for key, value in updates.items():
        setattr(action, key, value)


def build_argparser() -> argparse.ArgumentParser:
    parser = CE.build_argparser(description=__doc__)
    # Keep the shared parser/loop but remove its only LeRobot-specific required input.
    _set_base_argument(parser, "lerobot_dir", required=False, default=None,
                       help=argparse.SUPPRESS)
    _set_base_argument(
        parser, "episodes",
        help="seed slots, not dataset episode indices: '0-29' with --seed-base 100000 produces "
             "actual seeds 100000..100029")
    parser.add_argument("--task", required=True, help="registered RoboCasa environment/task name")
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE,
                        help=f"actual_seed = seed_base + episode slot (default {DEFAULT_SEED_BASE})")
    parser.add_argument("--scene-split", choices=["target", "pretrain", "all"], default="target",
                        help="RoboCasa layout/style and object-instance split (default target)")
    parser.add_argument("--robot", default="PandaOmron",
                        help="robot used to create the procedural environment (default PandaOmron)")
    parser.add_argument("--clutter-mode", type=int, default=1,
                        help="official target datasets use clutter_mode=1")
    parser.add_argument("--layout-id", type=int, default=None,
                        help="optional fixed layout; requires --style-id")
    parser.add_argument("--style-id", type=int, default=None,
                        help="optional fixed style; requires --layout-id")
    parser.add_argument("--randomize-cameras", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-reset-bundle", action=argparse.BooleanOptionalAction, default=True,
                        help="save initial state/model XML/ep_meta beside each rollout (default on)")
    parser.add_argument("--reset-only", action="store_true",
                        help="only instantiate the requested seeds and save reset bundles; no servers")
    parser.add_argument("--reset-bundle-root", type=Path, default=None,
                        help="reset-only output root; default <out-root>/procedural_resets")
    parser.add_argument(
        "--load-reset-root", type=Path, default=None,
        help="replay frozen bundles from <root>/<task>/seed_<10-digit seed>; recommended for "
             "all compared model arms")
    parser.add_argument(
        "--allow-live-reset", action="store_true",
        help="expert/debug option: evaluate a newly generated reset without a frozen bank; this "
             "is not safe for paired model comparison")
    parser.add_argument(
        "--overwrite-reset-bundles", action="store_true",
        help="replace complete reset-only bundles instead of verifying/reusing them")
    return parser


class ProceduralResetAdapter:
    """Provide the LeRobot-shaped inputs expected by combined_eval from a procedural reset."""

    def __init__(self, args):
        self.args = args
        self.env = None
        self.state: np.ndarray | None = None
        self.model_xml: str | None = None
        self.ep_meta: dict | None = None
        self.record: dict | None = None
        self.initial_images: dict[str, np.ndarray] | None = None

    def actual_seed(self, slot: int | None = None) -> int:
        slot = int(self.args.episode_index if slot is None else slot)
        seed = int(self.args.seed_base) + slot
        if not 0 <= seed < 2**32:
            raise ValueError(f"actual seed must fit uint32, got {seed}")
        return seed

    def _scene_kwargs(self) -> dict:
        if (self.args.layout_id is None) != (self.args.style_id is None):
            raise ValueError("--layout-id and --style-id must be supplied together")
        if self.args.layout_id is None:
            return {"split": self.args.scene_split}
        obj_split = self.args.scene_split if self.args.scene_split in ("target", "pretrain") else None
        return {
            "split": None,
            "obj_instance_split": obj_split,
            "layout_and_style_ids": [(self.args.layout_id, self.args.style_id)],
        }

    def bundle_dir(self, root: Path, slot: int | None = None) -> Path:
        return Path(root) / self.args.task / f"seed_{self.actual_seed(slot):010d}"

    def read_bundle(self, root: Path, slot: int | None = None, *, mark_replay: bool = True) -> dict:
        """Read and validate an exact procedural reset bundle."""
        bundle = self.bundle_dir(root, slot)
        missing = [name for name in _RESET_FILES if not (bundle / name).is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete reset bundle {bundle}; missing {missing}")

        source_record = json.loads((bundle / "reset.json").read_text())
        ep_meta = json.loads((bundle / "ep_meta.json").read_text())
        with np.load(bundle / "initial_state.npz", allow_pickle=False) as archive:
            state = np.asarray(archive["state"]).copy()
        with gzip.open(bundle / "model.xml.gz", "rt", encoding="utf-8") as f:
            model_xml = f.read()
        initial_images = {
            camera: np.asarray(Image.open(bundle / filename).convert("RGB")).copy()
            for camera, filename in _IMAGE_FILES.items()
        }

        expected_task = self.args.task
        expected_seed = self.actual_seed(slot)
        if source_record.get("task") != expected_task:
            raise ValueError(
                f"reset bundle task mismatch in {bundle}: "
                f"{source_record.get('task')!r} != {expected_task!r}")
        if int(source_record.get("seed", -1)) != expected_seed:
            raise ValueError(
                f"reset bundle seed mismatch in {bundle}: "
                f"{source_record.get('seed')!r} != {expected_seed}")
        expected_config = {
            "scene_split": self.args.scene_split,
            "robot": self.args.robot,
            "clutter_mode": int(self.args.clutter_mode),
            "randomize_cameras": bool(self.args.randomize_cameras),
        }
        mismatches = {
            field: {"bundle": source_record.get(field), "requested": expected}
            for field, expected in expected_config.items()
            if source_record.get(field) != expected
        }
        if self.args.layout_id is not None:
            for field, expected in (("layout_id", self.args.layout_id),
                                    ("style_id", self.args.style_id)):
                if source_record.get(field) != expected:
                    mismatches[field] = {
                        "bundle": source_record.get(field), "requested": expected}
        if mismatches:
            raise ValueError(f"reset bundle configuration mismatch in {bundle}: {mismatches}")

        state_hash = _sha256(np.ascontiguousarray(state).tobytes())
        xml_hash = _sha256(model_xml.encode("utf-8"))
        if state_hash != source_record.get("initial_state_sha256"):
            raise ValueError(f"initial-state checksum mismatch in {bundle}")
        if xml_hash != source_record.get("model_xml_sha256"):
            raise ValueError(f"model-XML checksum mismatch in {bundle}")
        image_hashes = source_record.get("initial_image_sha256", {})
        for camera, image in initial_images.items():
            image_hash = _sha256(np.ascontiguousarray(image).tobytes())
            if image_hash != image_hashes.get(camera):
                raise ValueError(f"initial-image checksum mismatch for {camera} in {bundle}")
        task_goal = (bundle / "task_goal.txt").read_text().strip()
        if task_goal != str(ep_meta.get("lang") or "").strip():
            raise ValueError(f"task_goal.txt does not match ep_meta['lang'] in {bundle}")

        self.state = state
        self.model_xml = model_xml
        self.ep_meta = ep_meta
        self.initial_images = initial_images
        self.record = dict(source_record)
        if mark_replay:
            self.record.update({
                "mode": "procedural_bundle_replay",
                "source_bundle_dir": str(bundle.resolve()),
                "source_mode": source_record.get("mode"),
                "runtime_robocasa_revision": _git_revision(
                    Path(__import__("robocasa").__file__).parents[1]),
            })
        return self.record

    def _make_uninitialized_env(self, seed: int):
        return create_env(
            env_name=self.args.task,
            robots=self.args.robot,
            camera_names=CAMERA_NAMES,
            camera_widths=256,
            camera_heights=256,
            seed=seed,
            clutter_mode=self.args.clutter_mode,
            randomize_cameras=self.args.randomize_cameras,
            translucent_robot=False,
            **self._scene_kwargs(),
        )

    def make_env(self, _unused_path=None):
        seed = self.actual_seed()
        with _isolated_reset_rng(seed):
            self.env = self._make_uninitialized_env(seed)
            if self.args.load_reset_root is not None:
                self.read_bundle(self.args.load_reset_root)
                _DATASET_RESET_TO(
                    self.env,
                    {
                        "states": self.state,
                        "model": self.model_xml,
                        "ep_meta": json.dumps(self.ep_meta),
                    },
                )
                return self.env
            obs = self.env.reset()
        self.ep_meta = self.env.get_ep_meta()
        instruction = str(self.ep_meta.get("lang") or "").strip()
        if not instruction:
            raise RuntimeError(f"{self.args.task} seed {seed} produced no task instruction")
        if bool(self.env._check_success()):
            raise RuntimeError(
                f"{self.args.task} seed {seed} is already successful at reset; reject this seed")
        camera_shapes = {}
        initial_images = {}
        for camera in CAMERA_NAMES:
            key = f"{camera}_image"
            if key not in obs:
                raise RuntimeError(f"{self.args.task} seed {seed} did not render {key}")
            shape = list(np.asarray(obs[key]).shape)
            if shape != [256, 256, 3]:
                raise RuntimeError(
                    f"{self.args.task} seed {seed} rendered {key} at {shape}, expected [256, 256, 3]")
            camera_shapes[camera] = shape
            # robosuite camera observations are bottom-up. Save the exact top-down RGB orientation
            # consumed by the policy and displayed by the evaluation GUI.
            image = np.asarray(obs[key])[::-1]
            if np.issubdtype(image.dtype, np.floating):
                image = (255 * image).astype(np.uint8)
            else:
                image = image.astype(np.uint8, copy=False)
            initial_images[camera] = np.ascontiguousarray(image)
        self.initial_images = initial_images
        self.state = np.asarray(self.env.sim.get_state().flatten()).copy()
        self.model_xml = self.env.sim.model.get_xml()
        state_bytes = np.ascontiguousarray(self.state).tobytes()
        xml_bytes = self.model_xml.encode("utf-8")
        self.record = {
            "schema": 2,
            "mode": "procedural_seed",
            "task": self.args.task,
            "seed_slot": int(self.args.episode_index),
            "seed": seed,
            "scene_split": self.args.scene_split,
            "layout_id": self.ep_meta.get("layout_id"),
            "style_id": self.ep_meta.get("style_id"),
            "obj_instance_split": (self.args.scene_split
                                   if self.args.scene_split in ("target", "pretrain") else None),
            "robot": self.args.robot,
            "clutter_mode": int(self.args.clutter_mode),
            "randomize_cameras": bool(self.args.randomize_cameras),
            "camera_names": list(CAMERA_NAMES),
            "camera_size": [256, 256],
            "camera_shapes": camera_shapes,
            "instruction": instruction,
            "initial_task_success": False,
            "rng_strategy": "env_seed+isolated_global_numpy+isolated_argumentless_default_rng",
            "replay_guarantee": "exact only when evaluated with --load-reset-root",
            "state_shape": list(self.state.shape),
            "initial_state_sha256": _sha256(state_bytes),
            "model_xml_sha256": _sha256(xml_bytes),
            "initial_image_sha256": {
                camera: _sha256(np.ascontiguousarray(image).tobytes())
                for camera, image in self.initial_images.items()
            },
            "robocasa_revision": _git_revision(Path(__import__("robocasa").__file__).parents[1]),
        }
        return self.env

    def episode_meta(self, _path, _episode_index) -> dict:
        if self.ep_meta is None:
            raise RuntimeError("procedural environment has not been reset")
        return self.ep_meta

    def episode_states(self, _path, _episode_index) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("procedural environment has not been reset")
        return self.state[None, :]

    def episode_model_xml(self, _path, _episode_index) -> str:
        if self.model_xml is None:
            raise RuntimeError("procedural environment has not been reset")
        return self.model_xml

    def reset_noop(self, env, _init) -> None:
        # make_env already performed exactly one reset. Assert that combined_eval is still operating
        # on that same object; silently ignoring a reset for another env would corrupt an evaluation.
        if env is not self.env:
            raise RuntimeError("procedural reset adapter received an unexpected environment")

    def write_bundle(self, root: Path) -> dict:
        if (self.record is None or self.state is None or self.model_xml is None
                or self.ep_meta is None or self.initial_images is None):
            raise RuntimeError("no procedural reset is available to save")
        root.mkdir(parents=True, exist_ok=True)
        CE._write_json(root / "reset.json", self.record)
        CE._write_json(root / "ep_meta.json", self.ep_meta)
        (root / "task_goal.txt").write_text(str(self.ep_meta["lang"]).strip() + "\n")
        np.savez_compressed(root / "initial_state.npz", state=self.state)
        with gzip.open(root / "model.xml.gz", "wb") as f:
            f.write(self.model_xml.encode("utf-8"))
        for camera, filename in _IMAGE_FILES.items():
            Image.fromarray(self.initial_images[camera]).save(root / filename)
        self._write_preview(root / "initial_views.png")
        return {
            **self.record,
            "bundle_dir": str(root),
            "files": list(_RESET_FILES),
        }

    def _write_preview(self, path: Path) -> None:
        """Write a human-readable three-camera contact sheet with the task goal."""
        if self.initial_images is None or self.ep_meta is None:
            raise RuntimeError("no procedural reset is available to preview")
        labels = [camera.removeprefix("robot0_") for camera in CAMERA_NAMES]
        goal_lines = textwrap.wrap(str(self.ep_meta["lang"]).strip(), width=108) or [""]
        header_height = 24
        footer_height = 14 * len(goal_lines) + 20
        canvas = Image.new("RGB", (256 * len(CAMERA_NAMES), 256 + header_height + footer_height),
                           "white")
        draw = ImageDraw.Draw(canvas)
        for i, (camera, label) in enumerate(zip(CAMERA_NAMES, labels, strict=True)):
            x = i * 256
            draw.text((x + 6, 5), label, fill="black")
            canvas.paste(Image.fromarray(self.initial_images[camera]),
                         (x, header_height))
        y = header_height + 256 + 7
        for line in goal_lines:
            draw.text((7, y), line, fill="black")
            y += 14
        canvas.save(path)


def _episode_dir(args, episode_id: str) -> Path:
    return Path(args.out_root) / args.method / episode_id.replace("/", "__")


def _minimal_reset_error(args, exc: Exception) -> dict:
    slot = int(args.episode_index)
    episode_id = f"{args.task}/{RESET_SPLIT_PREFIX}_{args.scene_split}/episode_{slot:06d}"
    doc = {
        "episode_id": episode_id,
        "task_name": args.task,
        "episode_success": False,
        "termination": "reset_error",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "procedural_reset": {
            "mode": ("procedural_bundle_replay" if args.load_reset_root
                     else "procedural_seed"),
            "task": args.task, "seed_slot": slot,
            "seed": int(args.seed_base) + slot, "scene_split": args.scene_split,
        },
    }
    CE._write_json(_episode_dir(args, episode_id) / "episode.json", doc)
    return doc


def _install_adapter(args, adapter: ProceduralResetAdapter) -> None:
    CE.make_camera_env = adapter.make_env
    CE.LU.get_episode_meta = adapter.episode_meta
    CE.LU.get_episode_states = adapter.episode_states
    CE.LU.get_episode_model_xml = adapter.episode_model_xml
    CE.reset_to = adapter.reset_noop
    CE._split_from_lerobot_dir = lambda _path: f"{RESET_SPLIT_PREFIX}_{args.scene_split}"

    original_eval = CE.eval_episode

    def eval_seeded(episode_dir, s1_client, s2_client, eval_args, out_root, method, norm_stats=None):
        adapter.env = None
        adapter.state = None
        adapter.model_xml = None
        adapter.ep_meta = None
        adapter.record = None
        adapter.initial_images = None
        try:
            doc = original_eval(
                episode_dir, s1_client, s2_client, eval_args, out_root, method, norm_stats)
        except Exception as exc:  # env construction/reset happens before combined_eval's try block
            if adapter.env is not None:
                try:
                    adapter.env.close()
                except Exception:
                    pass
            doc = _minimal_reset_error(eval_args, exc)
        if adapter.record is not None:
            ep_dir = _episode_dir(eval_args, doc["episode_id"])
            proc = dict(adapter.record)
            if eval_args.save_reset_bundle:
                proc = adapter.write_bundle(ep_dir / "reset")
            doc["procedural_reset"] = proc
            doc["lerobot_dir"] = None
            reset_mode = ("frozen procedural reset bundle" if eval_args.load_reset_root
                          else "live procedural RoboCasa seed")
            doc.setdefault("config", {})["reset_mode"] = reset_mode
            CE._write_json(ep_dir / "episode.json", doc)
        return doc

    CE.eval_episode = eval_seeded


def _procedural_config(args) -> dict:
    return {
        "mode": "procedural_bundle_replay" if args.load_reset_root else "procedural_seed",
        "seed_base": int(args.seed_base),
        "scene_split": args.scene_split,
        "robot": args.robot,
        "clutter_mode": int(args.clutter_mode),
        "layout_id": args.layout_id,
        "style_id": args.style_id,
        "randomize_cameras": bool(args.randomize_cameras),
        "camera_names": list(CAMERA_NAMES),
        "camera_size": [256, 256],
        "save_reset_bundle": bool(args.save_reset_bundle),
        "load_reset_root": (str(args.load_reset_root.resolve())
                            if args.load_reset_root is not None else None),
    }


def _guard_procedural_method(args) -> None:
    method_dir = Path(args.out_root) / args.method
    path = method_dir / "procedural_reset_config.json"
    cfg = _procedural_config(args)
    if path.exists():
        old = json.loads(path.read_text())
        if old != cfg:
            raise RuntimeError(
                f"procedural reset configuration mismatch in {path}: old={old}, new={cfg}. "
                "Use a distinct --method.")
    elif method_dir.exists() and any(method_dir.glob("*__episode_*")):
        raise RuntimeError(
            f"{method_dir} already contains episodes but no procedural reset guard; refusing to mix "
            "LeRobot and seeded-reset episodes. Use a distinct --method.")
    CE._write_json(path, cfg)


def run_reset_only(args, adapter: ProceduralResetAdapter) -> None:
    slots = CE._parse_episodes(args.episodes)
    root = args.reset_bundle_root or (Path(args.out_root) / "procedural_resets")
    manifest_path = root / args.task / "manifest.json"
    bank_config = {
        "scene_split": args.scene_split,
        "robot": args.robot,
        "clutter_mode": int(args.clutter_mode),
        "layout_id": args.layout_id,
        "style_id": args.style_id,
        "randomize_cameras": bool(args.randomize_cameras),
    }
    existing_records: dict[int, dict] = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("task") != args.task or int(previous.get("seed_base", -1)) != args.seed_base:
            raise RuntimeError(
                f"reset-bank manifest mismatch in {manifest_path}; use a distinct root for a "
                "different task/seed base")
        previous_config = previous.get("config")
        if previous_config is not None and previous_config != bank_config:
            raise RuntimeError(
                f"reset-bank configuration mismatch in {manifest_path}: "
                f"old={previous_config}, new={bank_config}")
        if previous_config is None and previous.get("scene_split") != args.scene_split:
            raise RuntimeError(f"reset-bank scene split mismatch in {manifest_path}")
        existing_records = {
            int(record["seed"]): record for record in previous.get("resets", [])
        }
    records = []
    generated = 0
    reused = 0
    for i, slot in enumerate(slots, 1):
        args.episode_index = slot
        bundle_dir = adapter.bundle_dir(root)
        present = [name for name in _RESET_FILES if (bundle_dir / name).is_file()]
        if present and len(present) != len(_RESET_FILES):
            missing = sorted(set(_RESET_FILES) - set(present))
            raise RuntimeError(
                f"refusing to overwrite partial reset bundle {bundle_dir}; missing {missing}")
        if len(present) == len(_RESET_FILES) and not args.overwrite_reset_bundles:
            record = adapter.read_bundle(root, mark_replay=False)
            records.append({**record, "bundle_dir": str(bundle_dir), "files": list(_RESET_FILES)})
            reused += 1
            print(f"[{i}/{len(slots)}] {args.task} seed={adapter.actual_seed()} "
                  f"layout={record['layout_id']} style={record['style_id']} REUSED", flush=True)
            continue
        env = None
        try:
            env = adapter.make_env()
            bundle = adapter.write_bundle(bundle_dir)
            records.append(bundle)
            generated += 1
            print(f"[{i}/{len(slots)}] {args.task} seed={adapter.actual_seed()} "
                  f"layout={bundle['layout_id']} style={bundle['style_id']} OK", flush=True)
        finally:
            if env is not None:
                env.close()
    for record in records:
        existing_records[int(record["seed"])] = record
    merged_records = [existing_records[seed] for seed in sorted(existing_records)]
    manifest = {
        "task": args.task, "seed_base": args.seed_base, "scene_split": args.scene_split,
        "config": bank_config,
        "n_resets": len(merged_records),
        "last_invocation": {
            "episode_slots": slots, "n_generated": generated, "n_reused": reused,
        },
        "resets": merged_records,
    }
    CE._write_json(manifest_path, manifest)
    print(f"WROTE {manifest_path} ({len(merged_records)} frozen resets total)", flush=True)


def main() -> None:
    args = build_argparser().parse_args()
    if args.reset_only and args.load_reset_root is not None:
        raise SystemExit("--reset-only generates a reset bank; do not combine it with "
                         "--load-reset-root")
    if args.overwrite_reset_bundles and not args.reset_only:
        raise SystemExit("--overwrite-reset-bundles is only valid with --reset-only")
    if not args.reset_only and args.load_reset_root is None and not args.allow_live_reset:
        raise SystemExit(
            "evaluation requires a frozen reset bank: generate it once with --reset-only "
            "--reset-bundle-root ROOT, then evaluate with --load-reset-root ROOT. For a "
            "non-comparable debug run only, pass --allow-live-reset.")
    args.lerobot_dir = str(_fake_episode_path(args.task))
    adapter = ProceduralResetAdapter(args)
    if args.reset_only:
        run_reset_only(args, adapter)
        return
    if not args.method:
        args.method = CE._short_method_name(args.s1_dir, args.s2_dir, "-seeded")
    _guard_procedural_method(args)
    _install_adapter(args, adapter)
    CE.run_sweep(args)


if __name__ == "__main__":
    main()
