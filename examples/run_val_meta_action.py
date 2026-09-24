# PAI-AV (clip_id, t0) 쌍마다 meta_action 결정과 궤적 예측을 함께 뽑아 쌍당 JSON 을 남기는 드라이버
"""Run Alpamayo 2 Super ``meta_action`` and ``trajectory`` on a manifest of (clip_id, t0_us).

Both tasks follow the released paths exactly and share one loaded 7-camera source:

meta_action  ``notebooks/meta_actions.ipynb``: ``select_task_input(.., "meta_action")`` →
             ``prepare_text_generation_inputs(task="meta_action")`` → ``generate_text``.
             Output: CoT text and the Longitudinal / Lateral / Lane meta-action text.
trajectory   ``inference_smoke``: ``select_task_input(.., "trajectory")`` →
             ``helper.prepare_model_inputs`` → ``sample_trajectories_from_data``.
             Output: one 64-step (6.4 s) predicted trajectory and its CoT.

The ground-truth future is recorded next to both, so the decision can be checked against
what the ego actually did and against the model's own predicted trajectory.

GPU runs require ``RUN_GPU=1``. Resume skips pairs whose JSON exists.
"""

import argparse
import collections
import concurrent.futures
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_val_autolabel as drv

from alpamayo2_super import helper
from alpamayo2_super.input_profiles import select_task_input
from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo2_super.text_tasks import generate_text, prepare_text_generation_inputs

SCHEMA_VERSION = "a2s_val_meta_action_v1"


def load_source(sample: dict, revision: str, retries: int) -> dict:
    """7-camera source ring for one pair. Runs in a worker thread."""
    for attempt in range(1, retries + 1):
        try:
            t_start = time.time()
            source = load_physical_aiavdataset(
                sample["clip_id"],
                t0_us=sample["t0_us"],
                avdi=drv._thread_avdi(revision),
                include_calibration=False,
            )
            return {"source": source, "load_s": round(time.time() - t_start, 2)}
        except drv.RETRYABLE as exc:
            if attempt == retries:
                raise
            drv.log(f"  [retry {attempt}/{retries}] {drv.stem(sample)} {type(exc).__name__}")
            drv._reset_thread_io()
            time.sleep(5 * 3 ** (attempt - 1))
    raise AssertionError("unreachable")


def summarize_future(xyz: np.ndarray) -> dict:
    """xyz: [64, 3] in the t0 frame (x forward, y left)."""
    p = np.vstack([np.zeros(3), xyz])
    speed = np.linalg.norm(np.diff(p, axis=0), axis=1) / 0.1
    return {
        "x64": round(float(xyz[-1, 0]), 2),
        "y64": round(float(xyz[-1, 1]), 2),
        "v_first": round(float(speed[0]), 2),
        "v_min": round(float(speed.min()), 2),
        "v_end": round(float(speed[-1]), 2),
    }


@torch.inference_mode()
def run_pair(model, source: dict, args) -> dict:
    out: dict[str, Any] = {}
    gt_xyz = source["ego_future_xyz"][0, 0].numpy()
    out["gt_future"] = summarize_future(gt_xyz)

    # meta_action (notebooks/meta_actions.ipynb)
    data = select_task_input(source, "meta_action")
    inputs = helper.to_device(
        prepare_text_generation_inputs(data, model.config, model.tokenizer, "meta_action"), "cuda"
    )
    torch.cuda.manual_seed_all(args.seed)
    t_start = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = generate_text(
            model, inputs, top_p=0.98, temperature=0.6, max_new_tokens=args.max_new_tokens
        )
    out["meta_action"] = {
        "cot": result["cot"][0],
        "meta_action": result["meta_action"][0],
        "raw_output": result["raw_outputs"][0],
        "generate_s": round(time.time() - t_start, 2),
    }
    del inputs

    # trajectory (inference_smoke)
    data = select_task_input(source, "trajectory")
    inputs = helper.to_device(
        helper.prepare_model_inputs(data, model.config, model.tokenizer), "cuda"
    )
    torch.cuda.manual_seed_all(args.seed)
    t_start = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _, _, extra = model.sample_trajectories_from_data(
            data=inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            diffusion_kwargs={"inference_step": args.diffusion_steps},
            return_extra=True,
        )
    pred = pred_xyz.float().cpu().numpy()[0, 0, 0]  # [64, 3]
    ade = float(np.linalg.norm(pred[:, :2] - gt_xyz[:, :2], axis=1).mean())
    fde = float(np.linalg.norm(pred[-1, :2] - gt_xyz[-1, :2]))
    out["trajectory"] = {
        "cot": str(extra["cot"].reshape(-1)[0]),
        "pred_future": summarize_future(pred),
        "pred_xyz": np.round(pred, 3).tolist(),
        "ade": round(ade, 3),
        "fde": round(fde, 3),
        "generate_s": round(time.time() - t_start, 2),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset-revision", default=drv.DEFAULT_DATASET_REVISION)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=512)  # generate_text's default
    ap.add_argument("--diffusion-steps", type=int, default=10)  # inference_smoke's default
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if os.environ.get("RUN_GPU") != "1":
        print("GPU run refused: set RUN_GPU=1.", file=sys.stderr)
        return 2
    torch.set_num_threads(16)
    drv.huggingface_hub.configure_http_backend(backend_factory=drv._timeout_session_factory)

    samples = drv.load_manifest(args.manifest)
    msha = drv.manifest_sha(samples)
    todo = samples[args.offset :]
    if args.limit is not None:
        todo = todo[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in todo if not (args.out_dir / f"{drv.stem(s)}.json").exists()]
    revision = drv.resolve_revision(args.dataset_revision)
    model_path = str(Path(args.model_path).resolve())
    drv.log(f"[meta_action] todo {len(todo)} | manifest sha {msha[:12]} | dataset {revision[:8]}")
    if not todo:
        return 0

    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    model = Alpamayo2Super.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0")
    common = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "model_revision": Path(model_path).name,
        "dataset_revision": revision,
        "manifest_sha": msha,
        "git_commit": drv.git_commit(),
        "driver_sha256": drv.hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    n_ok = n_err = 0
    t_run = time.time()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.prefetch)
    window: collections.deque = collections.deque()
    queue = iter(todo)

    def submit_next() -> None:
        sample = next(queue, None)
        if sample is not None:
            window.append((sample, pool.submit(load_source, sample, revision, args.retries)))

    try:
        for _ in range(args.prefetch):
            submit_next()
        index = 0
        while window:
            sample, future = window.popleft()
            submit_next()
            index += 1
            key = drv.stem(sample)
            record = {**common, "clip_id": sample["clip_id"], "t0_us": sample["t0_us"]}
            try:
                loaded = future.result(timeout=1800)
                record["load_s"] = loaded["load_s"]
                record.update(run_pair(model, loaded.pop("source"), args))
                record["finished_at_utc"] = drv.utc_now()
                drv.write_json_atomic(args.out_dir / f"{key}.json", record)
                n_ok += 1
                eta = (time.time() - t_run) / index * (len(todo) - index) / 60
                drv.log(
                    f"  [{index}/{len(todo)}] {key} | load {record['load_s']}s | "
                    f"{record['meta_action']['meta_action'][:60]!r} | "
                    f"ADE {record['trajectory']['ade']:.2f} | ETA {eta:.0f}m"
                )
            except Exception as exc:  # noqa: BLE001 — one bad pair must not stop the run
                n_err += 1
                record.update(
                    error_type=type(exc).__name__,
                    error=str(exc)[:2000],
                    traceback=traceback.format_exc()[-6000:],
                )
                drv.write_json_atomic(args.out_dir / "errors" / f"{key}.json", record)
                drv.log(f"  [{index}/{len(todo)}] {key} ERROR {type(exc).__name__}: {exc!s:.200}")
                if isinstance(exc, torch.AcceleratorError):
                    return 4
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        drv.log(
            f"[meta_action] finished | ok {n_ok} | error {n_err} | "
            f"{(time.time() - t_run) / 60:.1f} min"
        )
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
