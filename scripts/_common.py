from pathlib import Path
import argparse
import json
import sys
RUN_TIMESTAMP_KEYS = {
    'time', 'timestamp', 'timestamps', 'walltime', 'wall_time', 'wall_clock_time',
    'started_at', 'updated_at', 'completed_at', 'ended_at', 'stopped_at',
    'last_event_time', 'start_time', 'end_time', 'time_utc',
}


def strip_run_timestamps(value):
    """Keep steps, seeds and elapsed durations; omit wall-clock event times."""
    if isinstance(value, dict):
        return {key: strip_run_timestamps(item) for key, item in value.items()
                if key not in RUN_TIMESTAMP_KEYS}
    if isinstance(value, list):
        return [strip_run_timestamps(item) for item in value]
    return value

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def model_args(parser):
    from wordle_k64.model import MODEL_ID
    parser.add_argument("--model", help=f"Backbone ID or local snapshot (default: {MODEL_ID}, or checkpoint metadata)")
    parser.add_argument("--revision", help="Backbone revision; the released Qwen revision is pinned by default")
    parser.add_argument("--placement", choices=["two_gpu", "cuda:0", "cpu"], default="two_gpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--head-width", type=int, default=256)
    parser.add_argument("--head-seed", type=int, default=20261115)
    parser.add_argument("--architecture", choices=["char", "pointer", "lm_head"], default=None)
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--seed", type=int, default=20261112)


def build(args, checkpoint=None):
    from wordle_k64.model import build_policy
    return build_policy(args.model, args.placement, args.dtype, args.rank, args.alpha,
                        args.head_width, not args.no_grad_checkpointing, checkpoint,
                        not args.allow_download, head_seed=args.head_seed, architecture=args.architecture,
                        revision=args.revision)


def environment():
    from wordle_k64.environment import Engine
    from wordle_k64.sampler import ChallengeSampler
    from wordle_k64.contracts import load_contract
    split, env = load_contract()
    engine = Engine()
    sampler = ChallengeSampler(engine)
    if sampler.policy_id != env["candidate_policy_id"]:
        raise ValueError("Sampler policy ID changed")
    return split, engine, sampler


def log(out, **row):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    row = strip_run_timestamps(row)
    with (out / "log.jsonl").open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(row, allow_nan=False), flush=True)
    # JSONL retains all numeric diagnostics without timestamped event files.


def memory_metrics():
    import torch
    return {f"gpu_{i}": {
        "allocated_gib": torch.cuda.memory_allocated(i) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(i) / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(i) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(i) / 2**30,
    } for i in range(torch.cuda.device_count())}


def warmup(optimizer, steps):
    import torch
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: min(1.0, (n + 1) / max(1, steps)))


def positive(**values):
    for key, value in values.items():
        if value <= 0:
            raise ValueError(f"{key} must be positive")
