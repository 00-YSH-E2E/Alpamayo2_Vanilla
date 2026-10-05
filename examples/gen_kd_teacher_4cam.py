# A2 Super 를 로컬 PAD 4캠으로 돌려 KD teacher 캐시(CoC, K 궤적, x0, Euler velocity)와 평가 run 을 만드는 드라이버
"""Alpamayo 2 Super as a 4-camera teacher for the Alpamayo 1.5 3-camera student.

Cameras are [cross_left, front_wide, cross_right, front_tele] (ids 0,1,2,6), which is
not a released A2 profile; ``select_task_input`` is bypassed the same way the 1cam arm
of ``run_val_autolabel.py`` does. Data come from a local PAD root through the KD
repo's ``LocalPADInterface`` (never streamed).

The package samplers cannot take x0, cannot draw K trajectories from one CoC and do
not expose per-step velocities, so ``sample_teacher`` re-implements
``Alpamayo2Super.sample_trajectories_from_data`` with those three additions. With K=1
and no x0 the RNG is consumed in the same order (generate, then randn), so ``check``
mode can require bit-identical output against the package.

Modes:
  check  N pairs: package sample vs sample_teacher, same seed, must match exactly.
  eval   one t0 of a manifest, K=1, optional x0 replay from an Alpamayo 1.5 run's
         predictions.parquet; writes predictions/gt/run.json for compare_cam_kd.py.
  cache  every sample of a manifest (sharded): one CoC, K expert samples, x0, actions,
         xyz, Euler states/velocities and top-k logprobs, one .npz per sample.

GPU runs require ``RUN_GPU=1``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import einops
import numpy as np
import pandas as pd
import torch
from transformers import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

from alpamayo2_super import helper
from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo2_super.models.alpamayo2_super import (
    MaskDiscreteTrajectoryLogitsProcessor,
    _append_text_eos_mask,
)
from alpamayo2_super.models.expert_utils import (
    StopAfterEOS,
    build_expert_pos_ids_and_attn_mask,
    find_eos_offset,
    replace_padding_after_eos,
)
from alpamayo2_super.models.token_utils import extract_text_tokens
from alpamayo2_super.models.utils import fuse_traj_tokens

KD_REPO = Path(__file__).resolve().parents[2] / "YSH-KD-Alpamayo-1.5"
CAMERAS = ["camera_cross_left_120fov", "camera_front_wide_120fov",
           "camera_cross_right_120fov", "camera_front_tele_30fov"]
STUDENT_VOCAB = 155697  # Alpamayo 1.5 len(tokenizer); A2's lm_head is padded to 155776
TOPK = 20
DEFAULT_MODEL = ("/home/Humble/extra2/Model/hub/models--nvidia--Alpamayo2-Super/snapshots/"
                 "00554695e729a6ff0b6281fd2c81b18d06e33dbe")


def _load_local_pad():
    spec = importlib.util.spec_from_file_location(
        "kd_local_pad", KD_REPO / "src" / "alpamayo1_5" / "local_pad.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LocalPADInterface


def pair_seed(seed: int, clip_id: str, t0_us: int) -> int:
    digest = hashlib.sha256(f"{clip_id}:{t0_us}".encode()).digest()
    return (seed ^ int.from_bytes(digest[:4], "little")) & 0x7FFFFFFF


def load_samples(manifest: Path, t0_us: int | None = None) -> list[dict[str, Any]]:
    rows = json.loads(manifest.read_text())["samples"]
    out = [{"clip_id": str(r["clip_id"]), "t0_us": int(r["t0_us"])} for r in rows]
    return [s for s in out if t0_us is None or s["t0_us"] == t0_us]


def load_pair(avdi, sample: dict[str, Any], model) -> dict[str, Any]:
    source = load_physical_aiavdataset(
        sample["clip_id"], t0_us=sample["t0_us"], avdi=avdi, maybe_stream=False,
        camera_features=CAMERAS, include_calibration=False,
    )
    if source["camera_indices"].tolist() != [0, 1, 2, 6]:
        raise ValueError(f"unexpected camera order {source['camera_indices'].tolist()}")
    inputs = helper.prepare_model_inputs(source, model.config, model.tokenizer)
    return {"source": source, "inputs": inputs}


def _generate(model, tokenized_data, generation_config, **kwargs):
    """``Alpamayo2Super._generate_with_shared_prefill`` for one sequence, with logits kept."""
    input_ids = tokenized_data["input_ids"]
    mask = tokenized_data["attention_mask"]
    vision = {k: tokenized_data[k] for k in ("pixel_values", "image_grid_thw") if k in tokenized_data}
    prefill = model.vlm.model(input_ids=input_ids[:, :-1], attention_mask=mask[:, :-1],
                              use_cache=True, **vision)
    cache = prefill.past_key_values
    del prefill
    out = model.vlm.generate(input_ids=input_ids, attention_mask=mask, past_key_values=cache,
                             generation_config=generation_config, **kwargs)
    out.rope_deltas = model.vlm.model.rope_deltas.clone()
    return out


@torch.inference_mode()
def sample_teacher(model, data: dict[str, Any], *, k: int, x0: torch.Tensor | None = None,
                   top_p=0.98, temperature=0.6, max_new_tokens=256, inference_step=10,
                   keep_topk: bool = False) -> dict[str, Any]:
    """One CoC, then K expert samples from its KV, recording the Euler path."""
    tokenized = dict(data["tokenized_data"])
    traj_data = {"ego_history_xyz": data["ego_history_xyz"],
                 "ego_history_rot": data["ego_history_rot"]}
    tokenized["input_ids"] = fuse_traj_tokens(
        model.history_traj_tokenizer, model.future_traj_tokenizer, tokenized["input_ids"],
        traj_data, model.config.traj_ids)
    prompt_len = tokenized["input_ids"].shape[1]

    gen_cfg = copy.deepcopy(model.vlm.generation_config)
    gen_cfg.top_p, gen_cfg.temperature, gen_cfg.do_sample = top_p, temperature, True
    gen_cfg.max_new_tokens = max_new_tokens
    gen_cfg.num_return_sequences = 1
    gen_cfg.output_logits = keep_topk
    gen_cfg.return_dict_in_generate = True
    gen_cfg.top_k = None
    gen_cfg.pad_token_id = model.tokenizer.pad_token_id
    traj_offset = min(model.config.traj_ids["history_id0"], model.config.traj_ids["future_id0"])
    processors = LogitsProcessorList([MaskDiscreteTrajectoryLogitsProcessor(
        traj_token_offset=traj_offset, traj_vocab_size=model.config.traj_vocab_size)])
    eos_id = model.config.traj_ids["future_start"]
    _append_text_eos_mask(processors, gen_cfg.eos_token_id, preserved_token_id=eos_id)
    t_start = time.perf_counter()
    out = _generate(model, tokenized, gen_cfg, logits_processor=processors,
                    stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_id)]))
    out.sequences = replace_padding_after_eos(out.sequences, eos_id, model.tokenizer.pad_token_id)
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t_start

    cache = out.past_key_values
    cache.batch_repeat_interleave(k)
    seq_len = cache.get_seq_length()
    device = tokenized["input_ids"].device
    sequences = out.sequences.repeat_interleave(k, dim=0)
    offset = find_eos_offset(sequences=sequences, eos_token_id=eos_id, device=device)
    n_tok = model.expert.action_space.get_action_space_dims()[0]
    prefix_mask = tokenized["attention_mask"].repeat_interleave(k, dim=0)
    position_ids, attn_mask = build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=out.rope_deltas.repeat_interleave(k, dim=0),
        kv_cache_seq_len=seq_len, n_diffusion_tokens=n_tok, b_star=k, device=device,
        prefix_mask=prefix_mask)
    fwd = {"is_causal": False} if model.expert.config.expert_non_causal_attention else {}

    def step_fn(x, t):
        embeds = model.expert.action_in_proj(x, t)
        if embeds.dim() == 2:
            embeds = embeds.view(k, n_tok, -1)
        hidden = model.expert.expert(inputs_embeds=embeds, position_ids=position_ids,
                                     past_key_values=cache, attention_mask=attn_mask,
                                     use_cache=True, **fwd).last_hidden_state
        cache.crop(seq_len)
        return model.expert.action_out_proj(hidden).view(
            -1, *model.expert.action_space.get_action_space_dims())

    # FlowMatching._euler, with x0 injectable and every state/velocity kept.
    t_start = time.perf_counter()
    x_dims = model.expert.diffusion.x_dims
    x = torch.randn(k, *x_dims, device=device) if x0 is None else x0.to(device, torch.float32)
    times = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
    xs, vs = [x], []
    for i in range(inference_step):
        dt = (times[i + 1] - times[i]).view(1, 1, 1).expand(k, 1, 1)
        t_i = times[i].view(1, 1, 1).expand(k, 1, 1)
        v = step_fn(x=x, t=t_i)
        x = x + dt * v
        xs.append(x)
        vs.append(v)
    hist_xyz = einops.repeat(data["ego_history_xyz"][:, -1], "b ... -> (b n) ...", n=k)
    hist_rot = einops.repeat(data["ego_history_rot"][:, -1], "b ... -> (b n) ...", n=k)
    pred_xyz, pred_rot = model.expert.action_space.action_to_traj(x, hist_xyz, hist_rot)
    torch.cuda.synchronize()
    expert_s = time.perf_counter() - t_start

    generated = out.sequences[0, prompt_len:]
    eos_hits = (generated == eos_id).nonzero()
    eos_found = bool(eos_hits.numel())
    end = int(eos_hits[0, 0]) + 1 if eos_found else int(generated.numel())
    result = {
        "cot": extract_text_tokens(model.tokenizer, out.sequences)["cot"][0],
        "cot_ids": generated[:end].cpu().numpy().astype(np.int32),
        "eos_found": eos_found, "prompt_len": int(prompt_len),
        "x0": xs[0].float().cpu().numpy(), "action": x.float().cpu().numpy(),
        "xs": torch.stack(xs, 1).float().cpu().numpy(),
        "vs": torch.stack(vs, 1).float().cpu().numpy(),
        "xyz": pred_xyz.float().cpu().numpy(), "rot": pred_rot.float().cpu().numpy(),
        "gen_s": gen_s, "expert_s": expert_s,
    }
    if keep_topk:
        logits = torch.cat(out.logits[:end], dim=0).float()[:, :STUDENT_VOCAB]
        logits[:, traj_offset:traj_offset + model.config.traj_vocab_size] = float("-inf")
        top = torch.log_softmax(logits, dim=-1).topk(TOPK, dim=-1)
        result["topk_ids"] = top.indices.cpu().numpy().astype(np.int32)
        result["topk_logprob"] = top.values.cpu().numpy().astype(np.float16)
    return result


def iter_loaded(avdi, samples, model, workers: int):
    """Decode videos on CPU threads ahead of the GPU, keeping manifest order."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(load_pair, avdi, s, model) for s in samples[:workers * 2]]
        nxt = len(futures)
        for i, sample in enumerate(samples):
            loaded = futures[i].result()
            futures[i] = None
            if nxt < len(samples):
                futures.append(pool.submit(load_pair, avdi, samples[nxt], model))
                nxt += 1
            yield sample, loaded


def run_check(model, avdi, samples, args):
    for sample in samples[:args.n_check]:
        loaded = load_pair(avdi, sample, model)
        inputs = helper.to_device(loaded["inputs"], "cuda")
        seed = pair_seed(args.seed, sample["clip_id"], sample["t0_us"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref_xyz, _, _, extra = model.sample_trajectories_from_data(
                data=inputs, top_p=0.98, temperature=0.6, num_traj_samples=1,
                max_generation_length=256, diffusion_kwargs={"inference_step": 10},
                return_extra=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mine = sample_teacher(model, inputs, k=1)
        ref = ref_xyz.float().cpu().numpy().reshape(64, 3)
        same = np.array_equal(ref, mine["xyz"].reshape(64, 3))
        same_cot = str(extra["cot"].reshape(-1)[0]) == mine["cot"]
        print(f"[check] {sample['clip_id']}@{sample['t0_us']} xyz_equal={same} cot_equal={same_cot} "
              f"max_abs={np.abs(ref - mine['xyz'].reshape(64, 3)).max():.3g}", flush=True)
        if not (same and same_cot):
            return 1
    return 0


def x0_table(path: str | None) -> dict[str, np.ndarray] | None:
    if not path:
        return None
    table = pd.read_parquet(path, columns=["clip_id", "sample_k", "x0"])
    return {clip: np.stack([np.asarray(v, np.float32).reshape(64, 2)
                            for v in g.sort_values("sample_k")["x0"]])
            for clip, g in table.groupby("clip_id")}


def run_eval(model, avdi, samples, args, meta):
    x0s = x0_table(args.x0_from)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    preds, gts = [], []
    for sample, loaded in iter_loaded(avdi, samples, model, args.workers):
        inputs = helper.to_device(loaded["inputs"], "cuda")
        seed = pair_seed(args.seed, sample["clip_id"], sample["t0_us"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        x0 = None
        if x0s is not None:
            if sample["clip_id"] not in x0s:
                raise SystemExit(f"--x0-from has no x0 for {sample['clip_id']}")
            x0 = torch.as_tensor(x0s[sample["clip_id"]][:args.k])
        t_start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res = sample_teacher(model, inputs, k=args.k, x0=x0)
        total_ms = (time.perf_counter() - t_start) * 1e3
        for j in range(args.k):
            preds.append({
                "schema_version": 3, "variant": args.variant, "clip_id": sample["clip_id"],
                "t0_us": sample["t0_us"], "sample_k": j,
                "pred_xy": res["xyz"][j, :, :2].astype(np.float64).reshape(-1),
                "x0": res["x0"][j].astype(np.float64).reshape(-1),
                "cot": res["cot"], "token_ids": res["cot_ids"].astype(np.int64),
                "n_generated_tokens": int(len(res["cot_ids"])),
                "eos_missing": not res["eos_found"], "prompt_len": res["prompt_len"],
                "t_total_ms": total_ms, "model": "nvidia/Alpamayo2-Super",
                "data_spec": "PAD-Cam-4", "temperature": 0.6, "top_p": 0.98,
                "inference_step": 10, "num_traj_samples": args.k,
                "x0_source": "teacher" if x0 is not None else "sampled",
            })
        gt = loaded["source"]["ego_future_xyz"][0, 0, :, :2].numpy().astype(np.float64)
        gts.append({"clip_id": sample["clip_id"], "t0_us": sample["t0_us"], "gt_xy": gt.reshape(-1)})
        print(f"[eval] {len(gts)}/{len(samples)} {sample['clip_id']} {total_ms:.0f} ms "
              f"eos={res['eos_found']}", flush=True)
    pd.DataFrame(preds).to_parquet(out_dir / "predictions.parquet", index=False)
    pd.DataFrame(gts).to_parquet(out_dir / "gt.parquet", index=False)
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2))
    return 0


def run_cache(model, avdi, samples, args, meta):
    samples = samples[args.shard::args.num_shards]
    sample_dir = args.out / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out / f"index.shard{args.shard}of{args.num_shards}.jsonl"
    done = set()
    if index_path.exists():
        done = {(r["clip_id"], r["t0_us"]) for r in map(json.loads, index_path.read_text().splitlines())}
    todo = [s for s in samples if (s["clip_id"], s["t0_us"]) not in done]
    (args.out / f"run.shard{args.shard}of{args.num_shards}.json").write_text(json.dumps(meta, indent=2))
    print(f"[cache] shard {args.shard}/{args.num_shards}: {len(todo)} todo, {len(done)} done", flush=True)
    t_run = time.time()
    with index_path.open("a") as index:
        for i, (sample, loaded) in enumerate(iter_loaded(avdi, todo, model, args.workers), 1):
            inputs = helper.to_device(loaded["inputs"], "cuda")
            seed = pair_seed(args.seed, sample["clip_id"], sample["t0_us"])
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                res = sample_teacher(model, inputs, k=args.k, keep_topk=True)
            name = f"{sample['clip_id']}_{sample['t0_us']}.npz"
            tmp = sample_dir / (name + ".tmp.npz")
            np.savez(tmp, **{key: res[key] for key in (
                "cot_ids", "x0", "action", "xs", "vs", "xyz", "rot", "topk_ids", "topk_logprob")})
            os.replace(tmp, sample_dir / name)
            row = {"clip_id": sample["clip_id"], "t0_us": sample["t0_us"], "file": f"samples/{name}",
                   "seed": seed, "cot": res["cot"], "n_cot_ids": int(len(res["cot_ids"])),
                   "eos_found": res["eos_found"], "prompt_len": res["prompt_len"],
                   "gen_s": round(res["gen_s"], 3), "expert_s": round(res["expert_s"], 3)}
            index.write(json.dumps(row) + "\n")
            index.flush()
            if i % 10 == 0 or i == len(todo):
                rate = i / (time.time() - t_run)
                print(f"[cache] {i}/{len(todo)} {rate * 3600:.0f}/h "
                      f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["check", "eval", "cache"])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("/home/Humble/extra2/pad_1740_4cam"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--t0-us", type=int, default=None, help="eval: one t0 of the manifest")
    ap.add_argument("--x0-from", default=None, help="eval: predictions.parquet to replay x0 from")
    ap.add_argument("--variant", default="A2-4cam")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--n-check", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if os.environ.get("RUN_GPU") != "1":
        print("GPU run refused: set RUN_GPU=1.", file=sys.stderr)
        return 2
    if args.mode != "check" and args.out is None:
        ap.error("--out is required for eval and cache")
    if args.mode == "eval" and args.t0_us is None:
        ap.error("eval runs one t0 at a time (--t0-us), like the Alpamayo 1.5 runner")

    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    samples = load_samples(args.manifest, args.t0_us)[:args.limit]
    avdi = _load_local_pad()(args.data_root)
    t_start = time.time()
    model = Alpamayo2Super.from_pretrained(args.model_path, dtype=torch.bfloat16,
                                           device_map="cuda:0")
    model.eval()
    meta = {
        "mode": args.mode, "model_path": args.model_path, "cameras": CAMERAS,
        "manifest": str(args.manifest), "n_samples": len(samples), "t0_us": args.t0_us,
        "data_root": str(args.data_root), "data_lineage": avdi.lineage("pad_1740_4cam"),
        "k": args.k, "seed": args.seed, "seed_scheme": "seed xor sha256(clip:t0)[:4]",
        "top_p": 0.98, "temperature": 0.6, "top_k": None, "max_new_tokens": 256,
        "inference_step": 10, "x0_from": args.x0_from, "topk": TOPK,
        "student_vocab": STUDENT_VOCAB, "torch": torch.__version__,
        "attn_implementation": getattr(model.vlm.config, "_attn_implementation", None),
        "model_load_s": round(time.time() - t_start, 1),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    print(json.dumps({k: meta[k] for k in ("mode", "n_samples", "k", "attn_implementation",
                                           "model_load_s")}), flush=True)
    if args.mode == "check":
        return run_check(model, avdi, samples, args)
    if args.mode == "eval":
        return run_eval(model, avdi, samples, args, meta)
    return run_cache(model, avdi, samples, args, meta)


if __name__ == "__main__":
    sys.exit(main())
