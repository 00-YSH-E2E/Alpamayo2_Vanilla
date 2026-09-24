# A2 auto_labeling 이 미래 궤적(특히 좌우)을 실제로 읽는지 반사실 조건으로 확인하는 진단 스크립트
"""Probe whether Alpamayo 2 Super auto_labeling uses its future-trajectory condition.

One model load, two parts:

traj-sanity
    The official trajectory task (the same steps as ``inference_smoke``) on the checked-in
    manifests. A broken environment (attention, precision, library versions) shows up here
    as an unreasonable minADE.

counterfactual
    The same scene conditioned on three futures through the public
    ``prepare_text_generation_inputs(future_xyz=, future_rot=)`` argument:
      gt      the ground-truth future (reproduces the batch run)
      mirror  y -> -y and yaw -> -yaw in the t0 frame
      const   t0 speed held, straight
    Each is decoded with the official sampling (seed) and greedily. For pairs whose gt label
    names a maneuver direction, the logit margin logit(left) - logit(right) at that token is
    also measured under each future by teacher forcing the gt label prefix.

If the model reads lateral information, mirroring the future should move the margin and
flip the direction word. If it does not, the direction comes from the scene, not the
trajectory.

GPU runs require ``RUN_GPU=1``. ``--dry-run`` selects pairs, loads data and checks the
counterfactual futures by a tokenizer round trip, on CPU.
"""

import argparse
import concurrent.futures
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_val_autolabel as drv
from physical_ai_av import egomotion as em

from alpamayo2_super import helper
from alpamayo2_super.config import Alpamayo2SuperConfig
from alpamayo2_super.input_profiles import select_task_input
from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo2_super.models.utils import fuse_traj_tokens, tokenize_future_trajectory
from alpamayo2_super.text_tasks import (
    parse_auto_labeling_json,
    prepare_text_generation_inputs,
)

TASK = "auto_labeling"
FUTURES = ("gt", "mirror", "const")
DECODINGS = ("sample", "greedy")
LAT_VERB = re.compile(
    r"(nudg|lane change|chang\w* lanes?|turn|merg|shift|steer|veer|swerv|bias|move)\w*\s*"
    r"(?:slightly\s*|to\s+the\s*|toward\s+the\s*|into\s+the\s*|over\s+to\s+the\s*)*$",
    re.IGNORECASE,
)
LAT_TEXT = re.compile(LAT_VERB.pattern[:-1] + r"(left|right)\b", re.IGNORECASE)
STOP = re.compile(
    r"\b(?:stop|stops|stopped|stopping|halt|wait|waiting|yield|yielding)\b", re.IGNORECASE
)


# ---------------------------------------------------------------- pair selection (CPU)


def future_stats(interp, t0: int) -> tuple[float, float, float]:
    """Lateral offset at +6.4 s (t0 frame, + = left), minimum and initial speed over 0..6.4 s."""
    ts = np.array([t0 + k * 100_000 for k in range(65)], dtype=np.int64)
    e = interp(ts)
    xyz = e.pose.translation
    local = spt.Rotation.from_quat(e.pose.rotation.as_quat()[0]).inv().apply(xyz - xyz[0])
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / 0.1
    return float(local[-1, 1]), float(speed.min()), float(speed[0])


def coc_of(record: dict) -> str:
    return (record.get("auto_labeling_json") or {}).get("chain_of_causation") or ""


def select_pairs(args) -> list[dict]:
    clip_index = pd.read_parquet(args.pad_root / "clip_index.parquet")
    interps: dict[str, Any] = {}
    rows = []
    for path in sorted(args.a2_dir.glob("*_*.json")):
        if path.name.startswith("run_meta"):
            continue
        rec = json.loads(path.read_text())
        cid, t0 = rec["clip_id"], int(rec["t0_us"])
        if cid not in interps:
            chunk = int(clip_index.at[cid, "chunk"])
            zpath = args.pad_root / f"labels/egomotion/egomotion.chunk_{chunk:04d}.zip"
            with zipfile.ZipFile(zpath) as z:
                df = pd.read_parquet(io.BytesIO(z.read(f"{cid}.egomotion.parquet")))
            interps[cid] = em.EgomotionState.from_egomotion_df(df).create_interpolator(
                df["timestamp"].to_numpy(copy=True)
            )
        y64, vmin, v0 = future_stats(interps[cid], t0)
        m = LAT_TEXT.search(coc_of(rec))
        rows.append(
            {
                "clip_id": cid,
                "t0_us": t0,
                "y64": y64,
                "vmin": vmin,
                "v0": v0,
                "dir": m.group(m.lastindex).lower() if m else None,
                "raw_output": rec["raw_output"],
            }
        )
    d = pd.DataFrame(rows)
    # Half turns (|y| > 10 m), half small shifts (1..5 m: nudges and lane changes, where the
    # batch run went wrong), largest first within each.
    named = d[d.dir.notna()].sort_values("y64", key=abs, ascending=False)
    turns = named[named.y64.abs() > 10.0].head(args.n_lateral // 2)
    shifts = named[named.y64.abs().between(1.0, 5.0)].head(args.n_lateral - len(turns))
    lat = pd.concat([turns, shifts]).assign(kind="lateral")
    # Moving at t0 and stopping later, so the constant-speed future really removes the stop.
    stop = d[(d.vmin < 0.5) & (d.v0 > 3.0) & ~d.index.isin(lat.index)]
    stop = stop.sort_values(["clip_id", "t0_us"])
    stop = stop.head(args.n_stop).assign(kind="stop")
    return pd.concat([lat, stop]).to_dict("records")


# ---------------------------------------------------------------- counterfactual futures


def make_future(kind: str, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    xyz, rot = data["ego_future_xyz"].clone(), data["ego_future_rot"].clone()
    if kind == "gt":
        return xyz, rot
    if kind == "mirror":  # reflect across the x-z plane of the t0 frame
        flip = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
        return xyz * torch.tensor([1.0, -1.0, 1.0]), flip @ rot @ flip
    if kind == "const":
        hist = data["ego_history_xyz"][0, 0]
        v0 = float((hist[-1] - hist[-2]).norm() / 0.1)
        steps = torch.arange(1, xyz.shape[-2] + 1, dtype=xyz.dtype)
        const = torch.zeros_like(xyz)
        const[..., 0] = v0 * 0.1 * steps
        return const, torch.eye(3).expand_as(rot).clone()
    raise ValueError(kind)


def round_trip(tokenizer, data: dict, fxyz, frot) -> dict:
    """What the model actually receives: encode the future, decode it, summarize."""
    traj = {
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
        "ego_future_xyz": fxyz,
        "ego_future_rot": frot,
    }
    idx = tokenize_future_trajectory(tokenizer, traj)
    dec, _, _ = tokenizer.decode(traj["ego_history_xyz"][0], traj["ego_history_rot"][0], idx)
    dec = dec.reshape(-1, fxyz.shape[-2], 3)[0]
    p = torch.cat([torch.zeros(1, 3, dtype=dec.dtype), dec], 0)
    speed = (p[1:] - p[:-1]).norm(dim=-1) / 0.1
    return {
        "y64": round(float(dec[-1, 1]), 2),
        "x64": round(float(dec[-1, 0]), 1),
        "vmin": round(float(speed.min()), 2),
        "max_err_vs_input": round(float((dec - fxyz[0, 0]).norm(dim=-1).max()), 3),
    }


# ---------------------------------------------------------------- model-side probes


def direction_token(tokenizer, ids: list[int]) -> tuple[int, int, int] | None:
    """(position, left_id, right_id) of the first maneuver left/right in chain_of_causation."""
    text, in_coc = "", False
    for k, tid in enumerate(ids):
        piece = tokenizer.decode([tid])
        in_coc = in_coc or "chain_of_causation" in text
        word = piece.strip().lower()
        if in_coc and word in ("left", "right") and LAT_VERB.search(text[-60:]):
            tok = tokenizer.convert_ids_to_tokens(tid)
            other = tok.replace(word, "right" if word == "left" else "left")
            other_id = tokenizer.convert_tokens_to_ids(other)
            if other_id is None or other_id == tokenizer.unk_token_id:
                return None
            return (k, tid, other_id) if word == "left" else (k, other_id, tid)
        text += piece
    return None


@torch.inference_mode()
def direction_margin(
    model, task_inputs: dict, prefix_ids: list[int], left_id: int, right_id: int
) -> float:
    tok = dict(task_inputs["tokenized_data"])
    traj = {
        k: task_inputs[k]
        for k in ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")
    }
    ids = fuse_traj_tokens(
        model.history_traj_tokenizer,
        model.future_traj_tokenizer,
        tok["input_ids"],
        traj,
        model.config.traj_ids,
    )
    ids = torch.cat([ids, torch.tensor([prefix_ids], device=ids.device)], dim=1)
    extra = {k: v for k, v in tok.items() if k not in ("input_ids", "attention_mask")}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm(
            input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1, **extra
        )
    logits = out.logits[0, -1].float()
    return float(logits[left_id] - logits[right_id])


def generate(model, task_inputs, args, do_sample: bool) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gen = drv.generate_with_stats(
            model,
            task_inputs,
            top_p=args.top_p,
            top_k=None,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
        )
    text = gen["text"]
    try:
        coc = parse_auto_labeling_json(text).get("chain_of_causation") or ""
    except ValueError:
        coc = ""
    m = LAT_TEXT.search(coc)
    return {
        "coc": coc,
        "dir": m.group(m.lastindex).lower() if m else None,
        "stop": bool(STOP.search(coc)),
        "raw_output": gen["raw_output"],
        "ids": gen["generated_token_ids"],
        "n_tokens": gen["n_generated_tokens"],
    }


# ---------------------------------------------------------------- parts


def traj_sanity(model, args) -> list[dict]:
    samples = []
    for manifest in args.sanity_manifests:
        samples += json.loads(Path(manifest).read_text())["samples"]
    main_sha = drv.resolve_revision("main")  # inference_smoke streams from main

    def load(s):
        avdi = drv._thread_avdi(main_sha)  # one interface per worker thread
        src = load_physical_aiavdataset(s["clip_id"], t0_us=int(s["t0_us"]), avdi=avdi)
        return select_task_input(src, "trajectory")

    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for s, data in zip(samples, pool.map(load, samples)):
            inputs = helper.to_device(
                helper.prepare_model_inputs(data, model.config, model.tokenizer), "cuda"
            )
            torch.cuda.manual_seed_all(args.seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, _, _, extra = model.sample_trajectories_from_data(
                    data=inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    diffusion_kwargs={"inference_step": 10},
                    return_extra=True,
                )
            gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
            pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
            ade = float(np.linalg.norm(pred_xy - gt_xy[None], axis=1).mean(-1).min())
            fde = float(np.linalg.norm(pred_xy[:, :, -1] - gt_xy[None, :, -1], axis=1).min())
            row = {
                "clip_id": s["clip_id"],
                "t0_us": int(s["t0_us"]),
                "minADE": round(ade, 3),
                "FDE": round(fde, 3),
                "cot": str(extra["cot"].reshape(-1)[0]),
            }
            out.append(row)
            drv.log(
                f"  sanity {s['clip_id'][:8]} minADE {ade:.2f} m FDE {fde:.2f} m | "
                f"{row['cot'][:80]}"
            )
    return out


def counterfactual(model, args, pairs: list[dict], tokenizer_for_roundtrip, config) -> None:
    out_dir = args.out_dir
    revision = drv.resolve_revision(drv.DEFAULT_DATASET_REVISION)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.prefetch) as pool:
        futs = [
            pool.submit(
                drv.load_and_prepare,
                {"clip_id": p["clip_id"], "t0_us": p["t0_us"]},
                "1cam",
                revision,
                str(args.model_path),
                config,
                3,
                True,
            )
            for p in pairs
        ]
        for i, (pair, fut) in enumerate(zip(pairs, futs), 1):
            path = out_dir / f"cf_{pair['clip_id']}_{pair['t0_us']}.json"
            if path.exists():
                continue
            data = fut.result()["data"]
            rec = {k: pair[k] for k in ("clip_id", "t0_us", "kind", "y64", "vmin", "dir")}
            rec["futures"] = {}
            for kind in FUTURES:
                fxyz, frot = make_future(kind, data)
                entry = {"round_trip": round_trip(tokenizer_for_roundtrip, data, fxyz, frot)}
                if model is not None:
                    inputs = prepare_text_generation_inputs(
                        data, model.config, model.tokenizer, TASK, future_xyz=fxyz, future_rot=frot
                    )
                    inputs = helper.to_device(inputs, "cuda")
                    for dec in DECODINGS:
                        entry[dec] = generate(model, inputs, args, dec == "sample")
                    entry["_inputs"] = inputs
                rec["futures"][kind] = entry
            if model is not None:
                gt = rec["futures"]["gt"]
                rec["reproduces_batch"] = gt["sample"]["raw_output"] == pair["raw_output"]
                hit = direction_token(model.tokenizer, gt["sample"]["ids"])
                if hit:
                    pos, left_id, right_id = hit
                    prefix = gt["sample"]["ids"][:pos]
                    rec["margin"] = {
                        kind: round(
                            direction_margin(
                                model, rec["futures"][kind]["_inputs"], prefix, left_id, right_id
                            ),
                            3,
                        )
                        for kind in FUTURES
                    }
            for kind in FUTURES:
                rec["futures"][kind].pop("_inputs", None)
                for dec in DECODINGS:
                    rec["futures"][kind].get(dec, {}).pop("ids", None)
            drv.write_json_atomic(path, rec)
            f = rec["futures"]
            msg = " | ".join(
                f"{k}: y {f[k]['round_trip']['y64']:+.1f} v {f[k]['round_trip']['vmin']:.1f}"
                + (
                    f" → {f[k]['sample']['dir'] or '-'}/{f[k]['greedy']['dir'] or '-'}"
                    f"{' STOP' if f[k]['sample']['stop'] else ''}"
                    if "sample" in f[k]
                    else ""
                )
                for k in FUTURES
            )
            drv.log(
                f"  [{i}/{len(pairs)}] {pair['kind']:7s} {pair['clip_id'][:8]} {msg}"
                + (f" | margin {rec.get('margin')}" if "margin" in rec else "")
            )


def summarize(out_dir: Path) -> dict:
    recs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("cf_*.json"))]
    s: dict[str, Any] = {"n": len(recs)}
    s["reproduces_batch"] = float(np.mean([r.get("reproduces_batch", False) for r in recs]))
    lat = [r for r in recs if r["kind"] == "lateral"]
    for dec in DECODINGS:
        pairs = [(r["futures"]["gt"][dec]["dir"], r["futures"]["mirror"][dec]["dir"]) for r in lat]
        both = [(a, b) for a, b in pairs if a and b]
        s[f"lateral_{dec}_dir_flip_rate"] = (
            float(np.mean([a != b for a, b in both])) if both else None
        )
        s[f"lateral_{dec}_n_both_dir"] = len(both)
        s[f"lateral_{dec}_gt_dir_correct"] = float(
            np.mean(
                [
                    (r["futures"]["gt"][dec]["dir"] == "left") == (r["y64"] > 0)
                    for r in lat
                    if r["futures"]["gt"][dec]["dir"]
                ]
            )
        )
    margins = [r["margin"] for r in lat if "margin" in r]
    if margins:
        m_gt = np.array([m["gt"] for m in margins])
        m_mi = np.array([m["mirror"] for m in margins])
        m_co = np.array([m["const"] for m in margins])
        s["margin_n"] = len(margins)
        s["margin_sign_flip_rate_mirror"] = float(np.mean(np.sign(m_gt) != np.sign(m_mi)))
        s["margin_shift_toward_mirror_median"] = float(np.median((m_gt - m_mi) * np.sign(m_gt)))
        s["margin_abs_gt_median"] = float(np.median(np.abs(m_gt)))
        s["margin_const_minus_gt_median"] = float(np.median((m_co - m_gt) * np.sign(m_gt)))
    stop = [r for r in recs if r["kind"] == "stop"]
    for dec in DECODINGS:
        s[f"stop_{dec}_mention_gt"] = float(
            np.mean([r["futures"]["gt"][dec]["stop"] for r in stop])
        )
        s[f"stop_{dec}_mention_const"] = float(
            np.mean([r["futures"]["const"][dec]["stop"] for r in stop])
        )
        s[f"stop_{dec}_coc_unchanged_const"] = float(
            np.mean(
                [r["futures"]["gt"][dec]["coc"] == r["futures"]["const"][dec]["coc"] for r in stop]
            )
        )
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--a2-dir", type=Path, required=True, help="batch run dir (a2_1cam)")
    ap.add_argument("--pad-root", type=Path, default=Path("/home/Humble/extra2/pad_1740"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-lateral", type=int, default=20)
    ap.add_argument("--n-stop", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.98)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--prefetch", type=int, default=3)
    ap.add_argument(
        "--sanity-manifests",
        nargs="*",
        default=[
            "examples/public_golden_validation_samples.json",
            "examples/validation_samples.json",
        ],
    )
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and os.environ.get("RUN_GPU") != "1":
        print("GPU run refused: set RUN_GPU=1 (or pass --dry-run).", file=sys.stderr)
        return 2
    torch.set_num_threads(16)
    drv.huggingface_hub.configure_http_backend(backend_factory=drv._timeout_session_factory)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(args.model_path.resolve())
    config = Alpamayo2SuperConfig.from_pretrained(model_path)
    config._name_or_path = model_path

    pairs = select_pairs(args)
    drv.log(
        f"selected {sum(p['kind'] == 'lateral' for p in pairs)} lateral + "
        f"{sum(p['kind'] == 'stop' for p in pairs)} stop pairs"
    )
    # Round trips run on CPU with a separately built (weight-free) tokenizer, so they never
    # touch the model's device placement.
    import hydra.utils as hyu

    raw = json.loads((args.model_path / "config.json").read_text())
    rt_tokenizer = hyu.instantiate(raw["future_traj_tokenizer_cfg"], load_weights=False)
    if args.dry_run:
        args.out_dir = args.out_dir / "dryrun"  # never leave half records in the real out dir
        args.out_dir.mkdir(parents=True, exist_ok=True)
        counterfactual(None, args, pairs[:3] + pairs[-2:], rt_tokenizer, config)
        return 0

    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    t_start = time.time()
    model = Alpamayo2Super.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0")
    drv.log(f"model loaded in {time.time() - t_start:.0f}s")
    if not args.skip_sanity:
        rows = traj_sanity(model, args)
        drv.write_json_atomic(args.out_dir / "traj_sanity.json", {"rows": rows})
    counterfactual(model, args, pairs, rt_tokenizer, config)
    summary = summarize(args.out_dir)
    drv.write_json_atomic(args.out_dir / "summary.json", summary)
    drv.log("summary " + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
