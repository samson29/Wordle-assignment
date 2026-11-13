"""Frozen train-only complete games to define the fixed hard-secret pool."""
import argparse
import json
from pathlib import Path
import time
from _common import model_args, build, environment, log, memory_metrics
from wordle_k64.environment import seed_for
from wordle_k64.checkpoint import seed_everything
from wordle_k64.probe_study import ProbeFeedback, play_complete, records_from_games, write_json, digest


def main():
    p = argparse.ArgumentParser(); model_args(p)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--shards', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(2026111401, 2026111405)))
    args = p.parse_args(); started = time.monotonic(); seed_everything(args.seed)
    if args.shards < 1 or not 0 <= args.shard < args.shards or args.batch_size < 1:
        p.error('Require positive shard count/batch size and 0 <= shard < shards')
    if len(set(args.seeds)) != len(args.seeds):p.error('Mining seeds must be distinct')
    split, engine, sampler = environment(); feedback = ProbeFeedback(engine)
    assert not set(split['train']) & set(split['validation'] + split['test'])
    specs = [{'secret': secret, 'secret_index': i, 'mining_seed': seed,
              'environment_seed': seed_for('mining-environment', seed, secret),
              'action_seed': seed_for('mining-actions', seed, secret)}
             for seed in args.seeds for i, secret in enumerate(split['train'])]
    specs = specs[args.shard::args.shards]
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f'shard_{args.shard:02d}'
    dest.mkdir(exist_ok=True)
    policy = build(args, args.checkpoint); initial = digest(policy)
    identity = {'policy_sha256':initial, 'seeds':args.seeds, 'shard':args.shard,
                'shards':args.shards, 'batch_size':args.batch_size, 'temperature':1.2}
    if (dest/'identity.json').exists():
        if json.loads((dest/'identity.json').read_text()) != identity:
            raise ValueError('Mining output belongs to a different checkpoint or configuration')
    elif list(dest.glob('games_*.json')):
        raise ValueError('Existing mining games have no checkpoint identity; use a new output folder')
    write_json(dest/'identity.json',identity)
    log(dest, stage='initialization', step=0, total_games=len(specs), checkpoint=args.checkpoint,
        policy_sha256=initial, model_spec=policy.model_spec, batch_size=args.batch_size,
        note='Mining only: no gradients or importance ratios; batch8 recorded explicitly')
    totals = {'games':0, 'wins':0, 'six_turn_games':0, 'probe_required_games':0}
    for start in range(0, len(specs), 64):
        path = dest / f'games_{start:05d}.json'
        chunk = specs[start:start+64]
        if path.exists():
            records = json.loads(path.read_text())
            assert [{k:r[k] for k in s} for r,s in zip(records,chunk)] == chunk and len(records)==len(chunk)
        else:
            games, steps = play_complete(policy, engine, sampler, chunk, args.batch_size, 1.2)
            records = records_from_games(games, steps, chunk, feedback)
            write_json(path, records)
        totals['games'] += len(records)
        totals['wins'] += sum(r['won'] for r in records)
        totals['six_turn_games'] += sum(len(r['history'])==6 for r in records)
        totals['probe_required_games'] += sum(any(s['probe_eligible'] for s in r['steps']) for r in records)
        log(dest, stage='mining', step=totals['games'], total_games=len(specs), **totals,
            elapsed_s=time.monotonic()-started, gpu_hours=2*(time.monotonic()-started)/3600)
    assert digest(policy) == initial
    write_json(dest/'complete.json', {**totals, 'status':'complete', 'checkpoint':args.checkpoint,
        'policy_sha256':initial, 'training_updates':0, 'split':'train', 'batch_size':args.batch_size,
        'temperature':1.2, 'seeds':args.seeds, 'elapsed_s':time.monotonic()-started,
        'gpu_hours':2*(time.monotonic()-started)/3600, 'memory':memory_metrics()})


if __name__ == '__main__': main()
