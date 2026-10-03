#!/usr/bin/env python3
"""Convert franka_selfimit self-imitation ``.npz`` episodes into a LeRobot v2.1 dataset
GR00T's finetune pipeline can load directly.

Input (one file per kept episode, written by the G1/Franka self-imitation collector):
    seed<S>_ep<NNN>_<steps>steps.npz, keys:
        table_cam (T,200,200,3) uint8, wrist_cam (T,200,200,3) uint8,
        single_arm (T,9) f32, gripper (T,2) f32,
        action_single_arm (T,6) f32, action_gripper (T,1) f32, task (str)
    Row t is (observation BEFORE step t, action taken at step t) -- unchanged by this script,
    just repacked.

Output: a LeRobot v2.1 dataset tree (data/, videos/, meta/) using the plain pyarrow + ffmpeg
path -- no ``lerobot`` package dependency, since v2.1 is just parquet + jsonlines + json + mp4.
Schema verified against this repo's own sample datasets (demo_data/cube_to_bowl_5/meta/*) and
checkpoint-5000's experiment_cfg/conf.yaml (state/action delta_indices, modality_configs).

Video codec is h264 (libx264, -qp 0 lossless), not av1: gr00t/utils/video_utils.py's only
decoder is torchcodec.decoders.VideoDecoder, and this repo ships
examples/SimplerEnv/convert_av1_to_h264.py specifically to re-encode av1 clips to h264 before
training -- av1 is the problem case here, not the safe default.

fps=20: this env's control loop is sim.dt=1/200 (200Hz physics) at decimation=5 -> 20Hz control,
confirmed against both the original and current run's own printed step sizes (0.01 / 0.05s).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 20
CHUNKS_SIZE = 1000
NAME_RE = re.compile(r"^seed(\d+)_ep(\d+)_(\d+)steps\.npz$")

DATA_PATH_TEMPLATE = "data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{chunk:03d}/{video_key}/episode_{ep:06d}.mp4"
VIDEO_KEYS = ("table_cam", "wrist_cam")
# LeRobotEpisodeLoader resolves video paths from modality.json's video.<key>.original_key, not
# the short GR00T-facing name -- the on-disk directory/file naming must use that original key
# (observation.images.<cam>, same convention the sample datasets use), confirmed the hard way:
# ShardedSingleStepDataset.get_shard() raised "Could not open input file" for a path built from
# the short name until this was fixed.
VIDEO_ORIGINAL_KEY = {vk: f"observation.images.{vk}" for vk in VIDEO_KEYS}


def _encode_video(frames: np.ndarray, out_path: Path, fps: int = FPS) -> None:
    """frames: (T, H, W, 3) uint8 RGB. Pipes raw frames into ffmpeg -> lossless h264."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, c = frames.shape
    assert c == 3, f"expected 3-channel RGB frames, got shape {frames.shape}"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = proc.communicate(input=frames.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed encoding {out_path}: {stderr.decode(errors='replace')}")


def _stats_fingerprint(feature_name: str, dtype: str, shape: list[int]) -> str:
    """Byte-for-byte reimplementation of gr00t/data/stats.py's
    _compute_stats_fingerprint -- hashes only {feature, dtype, shape}, NOT data values. Must
    match exactly, or generate_stats() will treat a seeded stats.json as stale and silently
    recompute from this (narrower, self-imitation) dataset instead of reusing it."""
    payload = {"feature": feature_name, "dtype": dtype, "shape": shape}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def seed_stats_from_checkpoint(out_dir: Path, checkpoint_stats_path: Path, embodiment_tag: str) -> bool:
    """Pre-seed meta/stats.json with checkpoint-5000's own normalization ranges instead of
    letting generate_stats() compute fresh ones from this dataset.

    Why this matters (not just a nice-to-have): checkpoint-5000 learned a state/action space
    normalized by ITS training statistics (checkpoint_stats_path, keyed by embodiment_tag). A
    self-imitation set is narrower -- it's the policy's own successful outputs, not the original
    demonstration spread -- so letting generate_stats() recompute from it would shift the
    normalized space the pretrained action head was actually trained to decode, silently.

    How reuse is actually achieved: gr00t/data/stats.py's generate_stats() only recomputes a
    feature's stats if its __fingerprints__ entry is missing or doesn't match a hash of that
    feature's {dtype, shape} in info.json (gr00t/data/stats.py:183-198, :201-... -- the hash is
    schema-only, never touches data values). So a correctly-fingerprinted stats.json seeded here
    is accepted as fresh and reused verbatim -- confirmed by reading generate_stats() itself, not
    assumed. The seeded values flow into the finetuned checkpoint's own processor/statistics.json
    via pipeline.return_dataset() -> pipeline.return_processor() -> processor.save_pretrained
    (gr00t/experiment/experiment.py:248-256).

    checkpoint_stats_path's schema (verified directly, not guessed): top-level keyed by
    embodiment_tag, then {"state": {"single_arm": {...6 fields...}, "gripper": {...}},
    "action": {"single_arm": {...}, "gripper": {...}}} -- split by modality GROUP, 6 fields each
    (min/max/mean/std/q01/q99). meta/stats.json is keyed by flat LeRobot COLUMN name instead
    ("observation.state", "action") with each field as the full-width vector -- so this
    concatenates single_arm+gripper per field, in the same 0:9/9:11 and 0:6/6:7 order
    modality.json already declares.
    """
    if not checkpoint_stats_path.exists():
        print(f"[stats] {checkpoint_stats_path} not found -- leaving meta/stats.json unseeded "
              f"(generate_stats() will compute fresh from THIS dataset at train time)")
        return False

    with open(checkpoint_stats_path) as f:
        all_stats = json.load(f)
    if embodiment_tag not in all_stats:
        print(f"[stats] embodiment_tag={embodiment_tag!r} not in {checkpoint_stats_path} "
              f"(keys: {list(all_stats.keys())}) -- leaving meta/stats.json unseeded")
        return False

    src = all_stats[embodiment_tag]
    fields = ("min", "max", "mean", "std", "q01", "q99")

    def concat_group(top_key: str, sub_keys: tuple[str, ...]) -> dict:
        return {
            field: [v for sk in sub_keys for v in src[top_key][sk][field]]
            for field in fields
        }

    observation_state = concat_group("state", ("single_arm", "gripper"))
    action = concat_group("action", ("single_arm", "gripper"))
    assert len(observation_state["min"]) == 11, len(observation_state["min"])
    assert len(action["min"]) == 7, len(action["min"])

    stats = {
        "observation.state": observation_state,
        "action": action,
        "__fingerprints__": {
            "observation.state": _stats_fingerprint("observation.state", "float32", [11]),
            "action": _stats_fingerprint("action", "float32", [7]),
        },
    }
    stats_path = out_dir / "meta" / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"[stats] seeded meta/stats.json from {checkpoint_stats_path} "
          f"(embodiment_tag={embodiment_tag!r}) -- generate_stats() will reuse this, not "
          f"recompute from this dataset's own (narrower) data")
    return True


def _episode_table(d: dict, episode_index: int, task_index: int, index_start: int) -> pa.Table:
    """Builds the per-episode parquet table: observation.state(11)=single_arm(9)+gripper(2),
    action(7)=action_single_arm(6)+action_gripper(1), plus the standard LeRobot index columns."""
    t = d["single_arm"].shape[0]
    obs_state = np.concatenate([d["single_arm"], d["gripper"]], axis=1).astype(np.float32)
    action = np.concatenate([d["action_single_arm"], d["action_gripper"]], axis=1).astype(np.float32)
    assert obs_state.shape == (t, 11), obs_state.shape
    assert action.shape == (t, 7), action.shape

    table = pa.table({
        "observation.state": pa.array(list(obs_state)),
        "action": pa.array(list(action)),
        "timestamp": np.arange(t, dtype=np.float32) / FPS,
        "frame_index": np.arange(t, dtype=np.int64),
        "episode_index": np.full(t, episode_index, dtype=np.int64),
        "index": np.arange(index_start, index_start + t, dtype=np.int64),
        "task_index": np.full(t, task_index, dtype=np.int64),
    })
    return table


def convert(raw_dir: Path, out_dir: Path, max_steps: int, seed_stats_from: Path, embodiment_tag: str) -> list[dict]:
    npz_paths = sorted(raw_dir.glob("seed*_ep*_*steps.npz"))
    kept = []
    skipped = []
    for p in npz_paths:
        m = NAME_RE.match(p.name)
        if not m:
            print(f"[skip] unrecognized filename: {p.name}")
            continue
        steps = int(m.group(3))
        if steps > max_steps:
            skipped.append((p.name, steps))
            continue
        kept.append(p)

    if skipped:
        print(f"[filter] skipped {len(skipped)} episode(s) over --max_steps {max_steps}: "
              f"{[f'{n} ({s})' for n, s in skipped]}")
    if not kept:
        print("[filter] nothing to convert (no npz files, or all over --max_steps)")
        return []

    (out_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for vk in VIDEO_KEYS:
        (out_dir / "videos" / "chunk-000" / VIDEO_ORIGINAL_KEY[vk]).mkdir(parents=True, exist_ok=True)

    task_to_index: dict[str, int] = {}
    episode_records = []
    global_index = 0

    for episode_index, p in enumerate(kept):
        d = np.load(p, allow_pickle=True)
        task = str(d["task"])
        if task not in task_to_index:
            task_to_index[task] = len(task_to_index)
        task_index = task_to_index[task]

        t = d["single_arm"].shape[0]
        chunk = episode_index // CHUNKS_SIZE

        table = _episode_table(d, episode_index, task_index, global_index)
        pq.write_table(table, out_dir / DATA_PATH_TEMPLATE.format(chunk=chunk, ep=episode_index))

        for vk in VIDEO_KEYS:
            _encode_video(
                d[vk],
                out_dir / VIDEO_PATH_TEMPLATE.format(chunk=chunk, video_key=VIDEO_ORIGINAL_KEY[vk], ep=episode_index),
            )

        episode_records.append({
            "episode_index": episode_index,
            "tasks": [task],
            "length": t,
            "_source_file": p.name,
        })
        global_index += t
        print(f"[ok] episode {episode_index:06d} <- {p.name} ({t} steps, task_index={task_index})")

    write_meta(out_dir, episode_records, task_to_index, global_index)
    seed_stats_from_checkpoint(out_dir, seed_stats_from, embodiment_tag)
    return episode_records


def write_meta(out_dir: Path, episode_records: list[dict], task_to_index: dict[str, int], total_frames: int) -> None:
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task, idx in sorted(task_to_index.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for rec in episode_records:
            f.write(json.dumps({
                "episode_index": rec["episode_index"],
                "tasks": rec["tasks"],
                "length": rec["length"],
            }) + "\n")

    modality = {
        "state": {"single_arm": {"start": 0, "end": 9}, "gripper": {"start": 9, "end": 11}},
        "action": {"single_arm": {"start": 0, "end": 6}, "gripper": {"start": 6, "end": 7}},
        "video": {
            "table_cam": {"original_key": "observation.images.table_cam"},
            "wrist_cam": {"original_key": "observation.images.wrist_cam"},
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)

    total_episodes = len(episode_records)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka_panda",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "total_videos": total_episodes * len(VIDEO_KEYS),
        "total_chunks": max(1, -(-total_episodes // CHUNKS_SIZE)) if total_episodes else 0,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DATA_PATH_TEMPLATE.replace("{chunk:03d}", "{episode_chunk:03d}").replace("{ep:06d}", "{episode_index:06d}"),
        "video_path": VIDEO_PATH_TEMPLATE.replace("{chunk:03d}", "{episode_chunk:03d}").replace("{ep:06d}", "{episode_index:06d}"),
        "features": {
            "observation.state": {"dtype": "float32", "shape": [11],
                                   "names": [f"single_arm_{i}" for i in range(9)] + [f"gripper_{i}" for i in range(2)]},
            "action": {"dtype": "float32", "shape": [7],
                       "names": [f"single_arm_{i}" for i in range(6)] + ["gripper_0"]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    for vk in VIDEO_KEYS:
        info["features"][f"observation.images.{vk}"] = {
            "dtype": "video", "shape": [200, 200, 3], "names": ["height", "width", "channels"],
            "info": {
                "video.height": 200, "video.width": 200, "video.codec": "h264",
                "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                "video.fps": FPS, "video.channels": 3, "has_audio": False,
            },
        }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=str, default="/home/ubuntu/franka_selfimit/raw")
    parser.add_argument("--out_dir", type=str, default="/home/ubuntu/franka_selfimit/lerobot_dataset")
    parser.add_argument("--max_steps", type=int, default=540,
                         help="keep only episodes with <= this many steps (cleaner/faster successes)")
    parser.add_argument("--seed_stats_from", type=str,
                         default="/home/ubuntu/gr00t_checkpoint/statistics.json",
                         help="reuse this checkpoint's own normalization stats for meta/stats.json "
                              "instead of letting generate_stats() recompute from this (narrower) "
                              "self-imitation set; pass an empty string to disable")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    records = convert(
        Path(args.raw_dir), Path(args.out_dir), args.max_steps,
        Path(args.seed_stats_from) if args.seed_stats_from else Path("/nonexistent"),
        args.embodiment_tag,
    )
    print(f"\nConverted {len(records)} episode(s) -> {args.out_dir}")
    if not records:
        sys.exit(1)
