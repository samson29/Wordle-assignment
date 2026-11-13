"""Evaluate complete games with a learned readout, native zero-shot head or CPU heuristic."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from _common import ROOT, model_args, build, environment, log
from wordle_k64.baselines import NativeLabelPolicy, SolverPolicy
from wordle_k64.contracts import fingerprint
from wordle_k64.environment import Game, seed_for
from wordle_k64.probe_study import ProbeFeedback, write_json
from wordle_k64.probe_validation import evaluate_complete, cluster_ratio


def cpu_games(engine, sampler, secrets, seeds, mode, event=None):
    """Match the archived random-action and random-opening seed protocols."""
    heuristic = SolverPolicy(engine, sampler, 'consistent')
    records = []
    for seed in seeds:
        env_seeds = np.random.default_rng(seed).integers(2**63, size=len(secrets))
        for index, (secret, env_seed) in enumerate(zip(secrets, env_seeds)):
            action_rng = np.random.default_rng(seed_for('uniform-offered-action-v1', seed, index))
            opening_u = np.random.default_rng(seed_for('heuristic-opening-v1', seed, index)).random()
            game = Game(engine, sampler, secret, int(env_seed))
            steps = []
            while not game.done:
                decision = game.decision()
                if mode == 'random':
                    action = int(action_rng.integers(len(decision.words)))
                elif decision.history:
                    action = int(heuristic.score([decision]).argmax(-1)[0])
                elif mode == 'heuristic-salet':
                    action = decision.words.index('salet')
                else:
                    choices = sorted(w for w in decision.words
                                     if mode == 'heuristic-random' or len(set(w)) == 5)
                    word = choices[min(int(opening_u * len(choices)), len(choices) - 1)]
                    action = decision.words.index(word)
                result = game.step(action)
                steps.append({**decision.to_dict(), 'action': action,
                              'chosen_word': decision.words[action], 'feedback': result['feedback']})
            records.append({'seed': seed, 'answer_index': index, 'secret': secret,
                            'environment_seed': int(env_seed), 'won': bool(game.won),
                            'history': game.history, 'reward': game.terminal_reward(), 'steps': steps})
            if event and ((index+1) % 100 == 0 or index+1 == len(secrets)):
                event(seed=seed, completed=index+1, total=len(secrets))
    return records


def summarize_cpu(records, n_answers):
    draws = np.random.default_rng(722).integers(0, n_answers, size=(2000, n_answers))
    weights = np.array([np.bincount(d, minlength=n_answers) for d in draws], dtype=float)
    ids = [r['answer_index'] for r in records]
    wins = np.array([r['won'] for r in records])
    lengths = np.array([len(r['history']) for r in records])
    rate = cluster_ratio(wins, np.ones(len(records)), ids, weights)
    return {'games': len(records), 'unique_answers': n_answers, 'wins': int(wins.sum()),
            'win_rate': rate['mean'], 'answer_bootstrap_95pct': rate['answer_bootstrap_95pct'],
            'mean_guesses_wins': cluster_ratio(lengths * wins, wins, ids, weights),
            'guess_histogram': dict(Counter(str(len(r['history'])) if r['won'] else 'failed' for r in records)),
            'by_seed': [{'seed': s, 'games': sum(r['seed'] == s for r in records),
                         'wins': sum(r['won'] for r in records if r['seed'] == s)}
                        for s in sorted({r['seed'] for r in records})]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_args(parser)
    parser.add_argument('--policy', required=True, choices=[
        'checkpoint', 'zero-shot', 'random', 'heuristic-random', 'heuristic-distinct', 'heuristic-salet'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1001, 1002, 1003])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--limit', type=int, help='Mechanics smoke only; omit for final reporting')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--diagnostic-data', type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Output directory must be empty; keep completed evaluations immutable')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.batch_size < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Batch size must be positive and seeds must be distinct')
    if args.policy == 'checkpoint' and args.checkpoint is None:
        parser.error('--checkpoint is required for a learned policy')
    split, engine, sampler = environment()
    secrets = split[args.split][:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    protocol = {'policy': args.policy, 'checkpoint': str(args.checkpoint) if args.checkpoint else None,
                'split': args.split, 'seeds': args.seeds, 'contract': fingerprint(),
                'unique_answers': len(secrets), 'partial': args.limit is not None,
                'bootstrap_unit': 'answer, keeping seed repeats together', 'bootstrap_draws': 2000,
                'bootstrap_seed': 722, 'batch_size': args.batch_size}
    if args.policy in {'checkpoint', 'zero-shot'}:
        from wordle_k64.model import MODEL_ID, MODEL_REVISION
        from wordle_k64.contracts import sha256
        saved = json.loads((args.checkpoint/'metadata.json').read_text()) if args.checkpoint else {}
        model_id = args.model or saved.get('model_spec', {}).get('model_id', MODEL_ID)
        protocol.update(model=model_id, revision=args.revision or (MODEL_REVISION if model_id == MODEL_ID else None),
                        checkpoint_step=saved.get('progress', {}).get('step'),
                        checkpoint_iteration=saved.get('progress', {}).get('iteration'))
        if args.checkpoint:
            protocol['checkpoint_sha256'] = {name:sha256(args.checkpoint/name) for name in
                ['metadata.json','head.pt','adapters/policy/adapter_config.json','adapters/policy/adapter_model.safetensors']}
    write_json(args.out / 'protocol.json', protocol)
    if args.policy in {'random', 'heuristic-random', 'heuristic-distinct', 'heuristic-salet'}:
        records = cpu_games(engine, sampler, secrets, args.seeds, args.policy,
            event=lambda **kw:log(args.out,stage='evaluation_games',**kw))
        report = summarize_cpu(records, len(secrets))
        write_json(args.out / 'games.json', records)
    else:
        policy = (build(args, args.checkpoint) if args.policy == 'checkpoint' else
                  NativeLabelPolicy(args.model, args.placement, args.dtype, not args.allow_download, args.revision))
        panel = json.loads((ROOT / 'data/rl_turn5_panel.json').read_text())['states']
        report = evaluate_complete(policy, engine, sampler, secrets, args.seeds,
            ProbeFeedback(engine), panel, args.out, lambda stage, **kw: log(args.out, stage=stage, **kw),
            diagnostic_data=args.diagnostic_data, batch_size=args.batch_size)
    report.update(split=args.split, policy=args.policy, elapsed_s=time.monotonic() - started)
    write_json(args.out / 'metrics.json', report)
    print(json.dumps({k: report[k] for k in ['games', 'wins', 'win_rate', 'answer_bootstrap_95pct']}, indent=2))


if __name__ == '__main__':
    main()
