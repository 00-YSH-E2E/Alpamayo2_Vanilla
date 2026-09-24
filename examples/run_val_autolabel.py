# PAI-AV (clip_id, t0) 쌍마다 auto_labeling 을 돌려 쌍당 JSON 을 남기는 배치 드라이버
"""Batch driver for Alpamayo 2 Super ``auto_labeling`` over a manifest of (clip_id, t0_us).

The generation path follows ``notebooks/autolabeling.ipynb`` and
``alpamayo2_super.text_tasks.generate_text`` exactly, with one difference:
``generation_config.output_logits`` is turned on so that per-token log-probabilities
and entropies can be recorded. Recording logits does not touch sampling; run with
``--verify-official`` to compare the first pair against ``generate_text`` under the same
seed, and the driver-side tokenization against the notebook's (model.config/tokenizer).

Arms:
    6cam  load the canonical 7-camera ring, then ``select_task_input(..., "auto_labeling")``
          selects camera ids (0, 1, 2, 3, 5, 6). This is the released input profile.
    1cam  load ``camera_front_wide_120fov`` only and skip ``select_task_input`` (it requires
          the 7-camera source ring). The loader already computes the timing fields that
          ``select_input_profile`` would recompute, so the prompt is what a front-wide-only
          profile would produce. Visualization is not available for this arm.

Data are streamed from the Hugging Face dataset at a pinned revision. Upstream
``physical_ai_av`` streams chunk files from ``main`` whatever revision it was given, so
``PinnedAVDI`` passes the revision to the file system as well.

The manifest uses the same schema as ``examples/validation_samples.json``:
``{"samples": [{"clip_id": ..., "t0_us": ...}, ...]}``.

GPU runs require ``RUN_GPU=1``. ``--dry-run`` loads and tokenizes on CPU without the model.

Example:
    RUN_GPU=1 CUDA_VISIBLE_DEVICES=0 python examples/run_val_autolabel.py \\
        --arm 6cam --manifest /path/pairs_val347.json --out-root /path/alpamayo2 \\
        --model-path /path/Alpamayo2-Super/snapshots/<sha>
"""

import argparse
import collections
import concurrent.futures
import contextlib
import copy
import datetime
import gc
import hashlib
import http.client
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import huggingface_hub
import physical_ai_av
import torch
from transformers.generation.logits_process import LogitsProcessorList

from alpamayo2_super import helper
from alpamayo2_super.config import (
    Alpamayo2SuperConfig,
    build_alpamayo2_super_tokenizer,
)
from alpamayo2_super.input_profiles import select_task_input
from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo2_super.models.token_utils import extract_text_tokens
from alpamayo2_super.models.utils import fuse_traj_tokens
from alpamayo2_super.text_tasks import (
    parse_auto_labeling_json,
    prepare_text_generation_inputs,
    summarize_auto_labeling_conditioning,
)

SCHEMA_VERSION = "a2s_val_autolabel_v1"
TASK = "auto_labeling"
ARMS = ("6cam", "1cam")
DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
# The revision pad_1740 (and every Qwen CoC run) was built from. Its camera, egomotion and
# calibration chunk files for the val pairs hash-match main 33f9bf44 (2026-09-24 check).
DEFAULT_DATASET_REVISION = "b719eea7f0a63619ef51ec7f54178af0937ef050"
# Network and archive errors are retried; anything else (contract ValueError, KeyError,
# decode errors on intact bytes) is deterministic and is not.
RETRYABLE = (OSError, EOFError, zipfile.BadZipFile, http.client.HTTPException)
HTTP_TIMEOUT = (10, 120)  # connect, read — some huggingface_hub listing calls set none
# transformers treats these values as "unset" and substitutes the model's defaults.
SAMPLING_SENTINELS = {"temperature": 1.0, "top_p": 1.0, "top_k": 50}
# Settings that must match for a resume to add to an existing output directory.
RESUME_KEYS = (
    "schema_version",
    "arm",
    "future_source",
    "seed",
    "temperature",
    "top_p",
    "top_k",
    "max_new_tokens",
    "model_revision",
    "dataset_revision",
    "manifest_sha",
)
STATS_CHUNK = 128  # generation steps per GPU chunk when computing token statistics
MAX_CONSECUTIVE_ERRORS = 10
VERIFY_ATTEMPTS = 3

_tls = threading.local()
_print_lock = threading.Lock()


# ---------------------------------------------------------------- logging / io


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    with _print_lock:
        print(f"{utc_now()} {message}", flush=True)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    samples = json.loads(path.read_text(encoding="utf-8"))["samples"]
    out = [{"clip_id": str(s["clip_id"]), "t0_us": int(s["t0_us"])} for s in samples]
    keys = [(s["clip_id"], s["t0_us"]) for s in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"manifest has duplicate (clip_id, t0_us) keys: {path}")
    return out


def manifest_sha(samples: list[dict[str, Any]]) -> str:
    """Same fingerprint as pad-coc-protocol ``scoring/track_runs.py::manifest_sha``."""
    keys = sorted(f"{s['clip_id']}_{s['t0_us']}" for s in samples)
    return hashlib.sha1("\n".join(keys).encode()).hexdigest()


def stem(sample: dict[str, Any]) -> str:
    return f"{sample['clip_id']}_{sample['t0_us']}"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def driver_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ---------------------------------------------------------------- HTTP and dataset access


def _timeout_session_factory():
    """huggingface_hub's default session, with a default timeout on every request."""
    from huggingface_hub.utils import _http

    session = _http._default_backend_factory()
    request = session.request

    def request_with_timeout(method, url, *args, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = HTTP_TIMEOUT
        return request(method, url, *args, **kwargs)

    session.request = request_with_timeout
    return session


class PinnedAVDI(physical_ai_av.PhysicalAIAVDatasetInterface):
    """``PhysicalAIAVDatasetInterface`` whose streamed files come from ``self.revision``.

    Upstream ``open_file`` passes the revision to the cache lookup only; its streaming
    branch opens ``datasets/<repo>/<file>``, which resolves to ``main``.
    """

    @contextlib.contextmanager
    def open_file(self, filename: str, mode="rb", maybe_stream: bool = False):
        filepath = huggingface_hub.try_to_load_from_cache(
            filename=filename,
            cache_dir=self.cache_dir,
            **self.repo_snapshot_info,
        )
        if isinstance(filepath, str):
            with open(filepath, mode) as f:
                yield f
        elif maybe_stream:
            with self.fs.open(
                f"datasets/{self.repo_id}/{filename}", mode, revision=self.revision
            ) as f:
                yield f
        else:
            raise FileNotFoundError(f"{filename=} not found in cache and streaming is off")


def resolve_revision(revision: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    return huggingface_hub.HfApi().dataset_info(DATASET_REPO, revision=revision).sha


def _thread_avdi(revision: str) -> PinnedAVDI:
    if getattr(_tls, "avdi", None) is None:
        _tls.avdi = PinnedAVDI(revision=revision)
    return _tls.avdi


def _reset_thread_io() -> None:
    """Drop this thread's dataset interface so a retry starts with fresh HTTP state."""
    _tls.avdi = None
    huggingface_hub.HfFileSystem.clear_instance_cache()


def _thread_tokenizer(model_path: str, config: Alpamayo2SuperConfig):
    # The same construction Alpamayo2Super.__init__ uses. One per thread: a shared fast
    # tokenizer can raise "Already borrowed" when used from several threads at once.
    if getattr(_tls, "tokenizer", None) is None:
        _tls.tokenizer = build_alpamayo2_super_tokenizer(
            model_path, config.history_vocab_size, config.future_vocab_size
        )
    return _tls.tokenizer


def camera_features_for(arm: str, avdi: physical_ai_av.PhysicalAIAVDatasetInterface):
    if arm == "6cam":
        return None  # loader default = canonical 7-camera ring
    return [avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV]


def load_and_prepare(
    sample: dict[str, Any],
    arm: str,
    revision: str,
    model_path: str,
    config: Alpamayo2SuperConfig,
    retries: int,
    keep_data: bool = False,
) -> dict[str, Any]:
    """Load one pair and tokenize it on CPU. Runs in a worker thread."""
    key = stem(sample)
    for attempt in range(1, retries + 1):
        try:
            t_start = time.time()
            avdi = _thread_avdi(revision)
            source = load_physical_aiavdataset(
                sample["clip_id"],
                t0_us=sample["t0_us"],
                avdi=avdi,
                camera_features=camera_features_for(arm, avdi),
                # Calibration only feeds visualization overlays; it is not in the prompt.
                include_calibration=False,
            )
            data = select_task_input(source, TASK) if arm == "6cam" else source
            conditioning = summarize_auto_labeling_conditioning(data)
            t_load = time.time() - t_start

            t_start = time.time()
            task_inputs = prepare_text_generation_inputs(
                data=data,
                model_config=config,
                tokenizer=_thread_tokenizer(model_path, config),
                task=TASK,
            )
            loaded = {
                "task_inputs": task_inputs,
                "camera_indices": data["camera_indices"].tolist(),
                "camera_names": list(data["camera_names"]),
                "conditioning": conditioning,
                "prompt_tokens": int(task_inputs["tokenized_data"]["input_ids"].shape[1]),
                "load_s": round(t_load, 2),
                "prep_s": round(time.time() - t_start, 2),
                "load_attempts": attempt,
            }
            if keep_data:
                loaded["data"] = data
            return loaded
        except RETRYABLE as exc:
            if attempt == retries:
                raise
            log(f"  [retry {attempt}/{retries}] {key} {type(exc).__name__}: {str(exc)[:160]}")
            _reset_thread_io()
            time.sleep(5 * 3 ** (attempt - 1))
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- generation (main thread)


@torch.inference_mode()
def generate_with_stats(
    model: Any,
    data: dict[str, Any],
    top_p: float,
    top_k: int | None,
    temperature: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    """``text_tasks.generate_text`` for auto_labeling, plus token-level statistics.

    Identical to ``generate_text`` except ``output_logits=True``. Statistics are computed
    on the raw logits with the same trajectory-token mask applied and temperature 1,
    i.e. before temperature / top-p warping.
    """
    from alpamayo2_super.models.alpamayo2_super import MaskDiscreteTrajectoryLogitsProcessor

    if data["task"] != TASK:
        raise ValueError(f"expected task {TASK!r}, got {data['task']!r}")
    tokenized_data = dict(data["tokenized_data"])
    traj_data = {
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
        "ego_future_xyz": data["ego_future_xyz"],
        "ego_future_rot": data["ego_future_rot"],
    }
    tokenized_data["input_ids"] = fuse_traj_tokens(
        model.history_traj_tokenizer,
        model.future_traj_tokenizer,
        tokenized_data["input_ids"],
        traj_data,
        model.config.traj_ids,
    )
    prompt_length = tokenized_data["input_ids"].shape[1]

    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = max_new_tokens
    generation_config.output_logits = True  # the only difference from generate_text
    generation_config.return_dict_in_generate = True
    generation_config.top_k = top_k
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    traj_offset = min(model.config.traj_ids["history_id0"], model.config.traj_ids["future_id0"])
    traj_vocab_size = model.config.traj_vocab_size
    logits_processor = LogitsProcessorList(
        [
            MaskDiscreteTrajectoryLogitsProcessor(
                traj_token_offset=traj_offset,
                traj_vocab_size=traj_vocab_size,
            )
        ]
    )
    t_start = time.time()
    outputs = model.vlm.generate(
        **tokenized_data,
        generation_config=generation_config,
        logits_processor=logits_processor,
    )
    torch.cuda.synchronize()
    gen_s = time.time() - t_start

    t_start = time.time()
    generated_tokens = outputs.sequences[:, prompt_length:]
    extracted = extract_text_tokens(model.tokenizer, generated_tokens)
    text = extracted["cot_auto_labeling"][0] or extracted["cot"][0]

    # Token statistics on the GPU in chunks; only length-n vectors come back to the host.
    n_steps = len(outputs.logits)
    step_ids = generated_tokens[0, :n_steps]
    logprob_chunks, entropy_chunks = [], []
    with torch.autocast("cuda", enabled=False):
        for start in range(0, n_steps, STATS_CHUNK):
            chunk = torch.cat(outputs.logits[start : start + STATS_CHUNK], dim=0).float()
            chunk[:, traj_offset : traj_offset + traj_vocab_size] = float("-inf")
            logp = torch.log_softmax(chunk, dim=-1)
            ids = step_ids[start : start + STATS_CHUNK, None]
            logprob_chunks.append(logp.gather(1, ids).squeeze(1))
            entropy_chunks.append(torch.special.entr(logp.exp()).sum(dim=-1))
            del chunk, logp
    token_logprobs = torch.cat(logprob_chunks).cpu() if n_steps else torch.empty(0)
    token_entropies = torch.cat(entropy_chunks).cpu() if n_steps else torch.empty(0)
    step_ids = step_ids.cpu()
    del outputs, generated_tokens

    # The stop set generate() actually uses (model.vlm's config: 151645 only, see run_meta).
    eos_ids = generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
    eos_hit = bool(n_steps > 0 and int(step_ids[-1]) in eos_ids)
    lp_mean = float(token_logprobs.mean()) if n_steps else float("nan")
    return {
        "generated_token_ids": step_ids.tolist(),
        "text": text,
        "raw_output": extracted["raw_outputs"][0],
        "n_generated_tokens": n_steps,
        "eos_hit": eos_hit,
        "eos_missing": (not eos_hit) and n_steps >= max_new_tokens,
        "logprob_mean": lp_mean,
        "perplexity": math.exp(-lp_mean) if n_steps else float("nan"),
        "entropy_mean": float(token_entropies.mean()) if n_steps else float("nan"),
        "entropy_p95": float(torch.quantile(token_entropies, 0.95)) if n_steps else float("nan"),
        "token_logprobs": [round(float(v), 5) for v in token_logprobs],
        "token_entropies": [round(float(v), 5) for v in token_entropies],
        "generate_s": round(gen_s, 2),
        "postgen_s": round(time.time() - t_start, 2),
    }


def verify_prepare(model: Any, data: dict[str, Any], cpu_inputs: dict[str, Any]) -> dict[str, Any]:
    """Tokenize the notebook's way (model.config, model.tokenizer) and compare tensors."""
    reference = prepare_text_generation_inputs(data, model.config, model.tokenizer, TASK)
    ours, ref = cpu_inputs["tokenized_data"], reference["tokenized_data"]
    equal = {
        key: bool(key in ours and torch.equal(ours[key], ref[key]))
        for key in ref
        if isinstance(ref[key], torch.Tensor)
    }
    return {"prepare_equal": all(equal.values()), "prepare_equal_by_key": equal}


def verify_generation(
    model: Any, task_inputs: dict[str, Any], args: argparse.Namespace, ours: dict[str, Any]
) -> dict[str, Any]:
    """Re-run the released ``generate_text`` with the same seed and compare its output."""
    from alpamayo2_super.text_tasks import generate_text

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            official = generate_text(
                model,
                task_inputs,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
    except ValueError as exc:  # generate_text raises when its output has no JSON object
        return {"official_error": str(exc)[:500], "official_raw_equal": None}
    return {
        "official_raw_equal": official["raw_outputs"][0] == ours["raw_output"],
        "official_text_equal": official["cot_auto_labeling"][0] == ours["text"],
    }


# ---------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--model-path", required=True, help="local Alpamayo2-Super snapshot dir")
    ap.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.98)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--prefetch", type=int, default=3, help="concurrent load+tokenize workers")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--load-timeout", type=float, default=1800.0, help="seconds per pair load")
    ap.add_argument("--torch-threads", type=int, default=16)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="CPU only: load and tokenize")
    ap.add_argument(
        "--verify-official",
        action="store_true",
        help="compare the first pair with the notebook tokenization and generate_text",
    )
    args = ap.parse_args()
    for name, sentinel in SAMPLING_SENTINELS.items():
        if getattr(args, name) == sentinel:
            ap.error(
                f"--{name.replace('_', '-')} {sentinel} is treated as unset by "
                "transformers and silently replaced; use a nearby value"
            )
    return args


def check_resume(out_dir: Path, samples: list[dict[str, Any]], common: dict[str, Any]):
    """Split samples into done / todo. Refuse to mix records made with other settings."""
    done, todo, mismatched, driver_changed = [], [], [], 0
    for sample in samples:
        path = out_dir / f"{stem(sample)}.json"
        if not path.exists():
            todo.append(sample)
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            todo.append(sample)  # empty or torn file: regenerate
            continue
        diff = [k for k in RESUME_KEYS if record.get(k) != common[k]]
        if diff:
            mismatched.append((path.name, diff))
        else:
            done.append(sample)
            driver_changed += record.get("driver_sha256") != common["driver_sha256"]
    return done, todo, mismatched, driver_changed


def main() -> int:
    args = parse_args()
    if not args.dry_run and os.environ.get("RUN_GPU") != "1":
        print("GPU run refused: set RUN_GPU=1 (or pass --dry-run).", file=sys.stderr)
        return 2
    torch.set_num_threads(args.torch_threads)  # 64-thread OpenMP teams stall on a busy host
    huggingface_hub.configure_http_backend(backend_factory=_timeout_session_factory)

    model_path = str(Path(args.model_path).resolve())
    config = Alpamayo2SuperConfig.from_pretrained(model_path)
    config._name_or_path = model_path  # what from_pretrained(model) sets; get_processor reads it

    samples = load_manifest(args.manifest)
    msha = manifest_sha(samples)
    variant = f"a2_{args.arm}"
    out_dir = args.out_root / variant
    err_dir = out_dir / "errors"
    out_dir.mkdir(parents=True, exist_ok=True)
    revision = resolve_revision(args.dataset_revision)

    common = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "arm": args.arm,
        "future_source": "ground_truth",
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": True,
        "max_new_tokens": args.max_new_tokens,
        "model_revision": Path(model_path).name,
        "dataset_revision": revision,
        "manifest_sha": msha,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "driver_sha256": driver_sha256(),
    }

    selected = samples[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    done, todo, mismatched, driver_changed = check_resume(out_dir, selected, common)
    if mismatched:
        log(
            f"[{variant}] refusing to resume: {len(mismatched)} records in {out_dir} were made "
            f"with other settings, e.g. {mismatched[0]}. Use another --out-root."
        )
        return 5
    if driver_changed:
        log(
            f"[{variant}] note: {driver_changed} existing records came from another driver "
            "version (generation settings match)"
        )
    log(
        f"[{variant}] manifest {len(samples)} (sha {msha[:12]}) | done {len(done)} | "
        f"todo {len(todo)} | dataset {revision[:8]} | dry_run {args.dry_run}"
    )
    if not todo:
        return 0

    # Fail fast on auth / revision / network before loading 67 GiB of weights.
    t_start = time.time()
    probe = _thread_avdi(revision)
    probe.get_clip_feature(todo[0]["clip_id"], probe.features.LABELS.EGOMOTION, maybe_stream=True)
    log(f"[{variant}] dataset access ok ({time.time() - t_start:.1f}s)")

    model = None
    run_meta: dict[str, Any] = {
        **common,
        "task": TASK,
        "model_path": model_path,
        "manifest": str(args.manifest),
        "n_manifest": len(samples),
        "invoked_at_utc": utc_now(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "include_calibration": False,
    }
    if not args.dry_run:
        from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

        t_start = time.time()
        model = Alpamayo2Super.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda:0"
        )
        traj_offset = min(model.config.traj_ids["history_id0"], model.config.traj_ids["future_id0"])
        run_meta.update(
            {
                "model_load_s": round(time.time() - t_start, 1),
                "gpu_name": torch.cuda.get_device_name(0),
                "attn_implementation": getattr(model.vlm.config, "_attn_implementation", None),
                "token_facts": {
                    # generation_config.json in the snapshot is not applied to model.vlm.
                    "eos_token_ids_effective": model.vlm.generation_config.eos_token_id,
                    "pad_token_id": model.tokenizer.pad_token_id,
                    "traj_ids": dict(model.config.traj_ids),
                    "traj_mask_span": [traj_offset, traj_offset + model.config.traj_vocab_size],
                    "lm_head_rows": int(model.vlm.get_output_embeddings().weight.shape[0]),
                    "tokenizer_len": len(model.tokenizer),
                    "token_stats_basis": "pre-warp logits, traj span masked, T=1, nats",
                    "peak_mem_note": "includes the fp32 logits kept for statistics (<=0.6 GiB)",
                },
            }
        )
        log(
            f"[{variant}] model loaded in {run_meta['model_load_s']}s on "
            f"{run_meta['gpu_name']} attn={run_meta['attn_implementation']}"
        )
    stamp = run_meta["invoked_at_utc"].replace(":", "")
    write_json_atomic(
        out_dir / f"run_meta.{'dryrun.' if args.dry_run else ''}{stamp}.json", run_meta
    )

    counts = collections.Counter()
    consecutive_errors = 0
    verify_attempts = 0
    hung = False
    t_run = time.time()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.prefetch)
    window: collections.deque = collections.deque()
    queue = iter(enumerate(todo))

    def submit_next() -> None:
        item = next(queue, None)
        if item is None:
            return
        position, sample = item
        keep = args.verify_official and not args.dry_run and position == 0
        window.append(
            (
                sample,
                pool.submit(
                    load_and_prepare,
                    sample,
                    args.arm,
                    revision,
                    model_path,
                    config,
                    args.retries,
                    keep,
                ),
            )
        )

    try:
        for _ in range(args.prefetch):
            submit_next()
        index = 0
        while window:
            sample, future = window.popleft()
            submit_next()
            index += 1
            key = stem(sample)
            record: dict[str, Any] = {
                **common,
                "clip_id": sample["clip_id"],
                "t0_us": sample["t0_us"],
                "gen_ts": sample["t0_us"],
            }
            oom = fatal = False
            try:
                try:
                    loaded = future.result(timeout=args.load_timeout)
                except concurrent.futures.TimeoutError:
                    hung = True
                    raise TimeoutError(f"load exceeded {args.load_timeout}s") from None
                cpu_inputs = loaded.pop("task_inputs")
                data = loaded.pop("data", None)
                record.update(loaded)
                if args.dry_run:
                    counts["ok"] += 1
                    consecutive_errors = 0
                    log(
                        f"  [{index}/{len(todo)}] {key} dry ok | load {loaded['load_s']}s "
                        f"prep {loaded['prep_s']}s | prompt {loaded['prompt_tokens']} tok"
                    )
                    continue

                if data is not None:
                    record["verify_prepare"] = verify_prepare(model, data, cpu_inputs)
                    log(f"  verify_prepare: {record['verify_prepare']['prepare_equal']}")
                del data
                task_inputs = helper.to_device(cpu_inputs, "cuda")
                del cpu_inputs
                torch.cuda.reset_peak_memory_stats()
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)  # the notebook's seeding, per pair
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    gen = generate_with_stats(
                        model,
                        task_inputs,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                    )
                record["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if args.verify_official and verify_attempts < VERIFY_ATTEMPTS:
                    verify_attempts += 1
                    result = verify_generation(model, task_inputs, args, gen)
                    record["verify_official"] = result
                    if result.get("official_raw_equal") is not None:
                        verify_attempts = VERIFY_ATTEMPTS  # conclusive; stop verifying
                    log(f"  verify_official: {result}")
                del task_inputs
                text = gen.pop("text")
                record.update(gen)
                record["auto_labeling_text"] = text
                try:
                    record["auto_labeling_json"] = parse_auto_labeling_json(text)
                    record["parse_error"] = None
                except ValueError as exc:
                    record["auto_labeling_json"] = None
                    record["parse_error"] = str(exc)
                record["finished_at_utc"] = utc_now()
                write_json_atomic(out_dir / f"{key}.json", record)
                stale = err_dir / f"{key}.json"
                if stale.exists():
                    stale.unlink()
                counts["ok"] += 1
                counts["parse_fail"] += record["parse_error"] is not None
                counts["eos_missing"] += record["eos_missing"]
                consecutive_errors = 0
                elapsed = time.time() - t_run
                eta_min = elapsed / index * (len(todo) - index) / 60
                coc = (record["auto_labeling_json"] or {}).get("chain_of_causation") or ""
                log(
                    f"  [{index}/{len(todo)}] {key} | load {record['load_s']}s "
                    f"gen {record['generate_s']}s post {record['postgen_s']}s | "
                    f"{record['n_generated_tokens']} tok eos={record['eos_hit']} | "
                    f"peak {record['peak_mem_gib']}GiB | "
                    f"parse={'ok' if record['parse_error'] is None else 'FAIL'} | "
                    f"ETA {eta_min:.0f}m | {coc[:70]}"
                )
            except Exception as exc:  # noqa: BLE001 — one bad pair must not stop 347
                counts["error"] += 1
                consecutive_errors += 1
                oom = isinstance(exc, torch.cuda.OutOfMemoryError)
                fatal = isinstance(exc, torch.AcceleratorError)  # sticky CUDA error
                record["error_type"] = type(exc).__name__
                record["error"] = str(exc)[:2000]
                record["traceback"] = traceback.format_exc()[-6000:]
                record["finished_at_utc"] = utc_now()
                write_json_atomic(err_dir / f"{key}.json", record)
                log(f"  [{index}/{len(todo)}] {key} ERROR {type(exc).__name__}: {str(exc)[:200]}")
            if oom:  # after the except block, so the traceback no longer pins GPU memory
                task_inputs = gen = None
                gc.collect()
                torch.cuda.empty_cache()
            if fatal:
                log(f"[{variant}] aborting: CUDA error is not recoverable in this process")
                return 4
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log(f"[{variant}] aborting: {consecutive_errors} consecutive errors")
                return 4
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        log(
            f"[{variant}] finished | ok {counts['ok']} | error {counts['error']} | "
            f"parse_fail {counts['parse_fail']} | eos_missing {counts['eos_missing']} | "
            f"{(time.time() - t_run) / 60:.1f} min"
        )
        if hung:  # a worker stuck in I/O would block interpreter exit
            sys.stdout.flush()
            os._exit(1)
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
