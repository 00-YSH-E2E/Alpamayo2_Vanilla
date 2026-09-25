# 결정을 먼저 주고 설명만 쓰게 하면(결정 → 설명 두 단계) A2 의 CoC 가 행동과 맞는지 VQA 로 확인하는 파일럿
"""Two-stage probe: give Alpamayo 2 Super a driving decision, then ask only for the reason.

Released auto_labeling writes decision and reason in one pass, and the reason does not
follow the ego motion (left/right at chance). This probe splits the job: the decision
comes first, and the model's VQA mode (the Cosmos reasoner backbone, no trajectory head)
is asked to explain it in one Chain-of-Causation sentence.

Three questions per pair, all through the released VQA path
(``select_task_input(.., "vqa")`` → ``prepare_vqa_inputs`` → ``generate_text``):

  none    no decision given (control: only the task format changes)
  a2meta  the decision A2 itself predicted (``run_val_meta_action.py`` output)
  truth   the decision the ego actually made, written in A2's meta-action vocabulary from a
          physical classification of the ground-truth future (heading change, lateral shift,
          speed profile)

GPU runs require ``RUN_GPU=1``. ``--dry-run`` prints the questions without the model.
"""

import argparse
import collections
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_val_autolabel as drv
import run_val_meta_action as mav

from alpamayo2_super import helper
from alpamayo2_super.input_profiles import select_task_input
from alpamayo2_super.text_tasks import generate_text, prepare_vqa_inputs

QUESTIONS = ("none", "a2meta", "truth")
ASK = (
    "Write one Chain-of-Causation sentence for this decision: state the driving action, then the "
    "specific object, agent, or road condition in the scene that causes it. Answer with that one "
    "sentence only."
)
ASK_NONE = (
    "What is the ego vehicle's driving decision over the next few seconds? Write one "
    "Chain-of-Causation sentence: state the driving action, then the specific object, agent, or "
    "road condition in the scene that causes it. Answer with that one sentence only."
)


def side(value: float) -> str:
    return "Left" if value > 0 else "Right"  # t0 frame: y and heading positive = left


def truth_decision(row: pd.Series) -> str:
    """Physical classes of the actual future → A2's meta-action vocabulary."""
    plon, plat = str(row.plon), str(row.plat)
    dv = float(row.dv)
    if plon.startswith(("0_", "1_", "2_", "4_")):
        lon = "Stop"
    elif plon.startswith("5_"):
        lon = "Strong Deceleration" if dv <= -3.0 else "Gentle Deceleration"
    elif plon.startswith(("3_", "7_")):
        lon = "Strong Acceleration" if dv >= 3.0 else "Gentle Acceleration"
    else:
        lon = "Maintain Speed"
    dpsi, y_end = float(row.dpsi_deg), float(row.y_end)
    shift = row.r_max_signed if pd.notna(row.get("r_max_signed")) else y_end
    if plat.startswith("1_"):
        lat, lane = f"Sharp Steer {side(dpsi)}", f"Turn {side(dpsi)}"
    elif plat.startswith("2_"):
        lat, lane = f"Steer {side(y_end)}", f"{side(y_end)} Lane Change"
    elif plat.startswith("4_"):
        lat, lane = f"Steer {side(float(shift))}", f"Slightly Shift {side(float(shift))}"
    elif plat.startswith("3_") or (plat.startswith("6_") and abs(dpsi) >= 5.0):
        lat, lane = f"Steer {side(dpsi)}", "Lane Keep"
    else:
        lat, lane = "Go Straight", "Lane Keep"
    return f"Longitudinal: {lon}. Lateral: {lat}. Lane: {lane}."


def one_line(meta_action: str) -> str:
    return " ".join(line.strip() for line in meta_action.splitlines() if line.strip())


def build_questions(args) -> list[dict]:
    phys = pd.read_parquet(args.physics).set_index(["clip_id", "gen_ts"])
    items = []
    for path in sorted(args.pairs_dir.glob("cf_*.json")):
        pair = json.loads(path.read_text())
        key = (pair["clip_id"], int(pair["t0_us"]))
        meta = json.loads((args.meta_dir / f"{key[0]}_{key[1]}.json").read_text())
        decisions = {
            "a2meta": one_line(meta["meta_action"]["meta_action"]),
            "truth": truth_decision(phys.loc[key]),
        }
        questions = {"none": ASK_NONE}
        for name, decision in decisions.items():
            questions[name] = (
                f"The ego vehicle's driving decision over the next few seconds is: {decision} {ASK}"
            )
        items.append(
            {
                "clip_id": key[0],
                "t0_us": key[1],
                "kind": pair["kind"],
                "decisions": decisions,
                "questions": questions,
            }
        )
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = Path("/home/Humble/extra2/pad_protocol/alpamayo2")
    ap.add_argument("--pairs-dir", type=Path, default=root / "probe")
    ap.add_argument("--meta-dir", type=Path, default=root / "meta_action")
    ap.add_argument(
        "--physics",
        type=Path,
        default=root / "analysis/behavior_label_validation/merged_val4.parquet",
    )
    ap.add_argument("--out-dir", type=Path, default=root / "two_stage")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset-revision", default=drv.DEFAULT_DATASET_REVISION)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = build_questions(args)
    drv.log(f"[two_stage] {len(items)} pairs")
    if args.dry_run:
        for item in items[:4] + items[-2:]:
            print(f"\n{item['kind']} {item['clip_id'][:8]} {item['t0_us']}")
            for name in QUESTIONS[1:]:
                print(f"  {name:6s} {item['decisions'][name]}")
        return 0
    if os.environ.get("RUN_GPU") != "1":
        print("GPU run refused: set RUN_GPU=1.", file=sys.stderr)
        return 2
    torch.set_num_threads(16)
    drv.huggingface_hub.configure_http_backend(backend_factory=drv._timeout_session_factory)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in items if not (args.out_dir / f"{i['clip_id']}_{i['t0_us']}.json").exists()]
    revision = drv.resolve_revision(args.dataset_revision)

    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    model = Alpamayo2Super.from_pretrained(
        str(Path(args.model_path).resolve()), dtype=torch.bfloat16, device_map="cuda:0"
    )
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.prefetch)
    window: collections.deque = collections.deque()
    queue = iter(todo)

    def submit_next() -> None:
        item = next(queue, None)
        if item is not None:
            window.append((item, pool.submit(mav.load_source, item, revision, 3)))

    t_run = time.time()
    try:
        for _ in range(args.prefetch):
            submit_next()
        index = 0
        while window:
            item, future = window.popleft()
            submit_next()
            index += 1
            data = select_task_input(future.result(timeout=1800)["source"], "vqa")
            item["answers"] = {}
            for name in QUESTIONS:
                inputs = helper.to_device(
                    prepare_vqa_inputs(
                        data, model.config, model.tokenizer, item["questions"][name]
                    ),
                    "cuda",
                )
                torch.cuda.manual_seed_all(args.seed)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    result = generate_text(
                        model,
                        inputs,
                        top_p=0.98,
                        temperature=0.6,
                        max_new_tokens=args.max_new_tokens,
                    )
                item["answers"][name] = {
                    "answer": result["answer"][0],
                    "raw_output": result["raw_outputs"][0],
                }
            item["finished_at_utc"] = drv.utc_now()
            drv.write_json_atomic(args.out_dir / f"{item['clip_id']}_{item['t0_us']}.json", item)
            drv.log(
                f"  [{index}/{len(todo)}] {item['kind']} {item['clip_id'][:8]} | "
                + " | ".join(f"{n}: {item['answers'][n]['answer'][:60]!r}" for n in QUESTIONS)
            )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        drv.log(f"[two_stage] finished in {(time.time() - t_run) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
