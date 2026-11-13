"""Recompute top-option ECE from archived complete games and public solver labels.

CPU only; reads the evaluation archive without changing it. Confidence is the
original, unscaled probability of the greedy action, not its solver win value.
All tied maximal solver values count as correct. The closing-aware label also
requires an immediate correct guess when public feedback identifies one answer.
Solver values before turn 5 use a fixed continuation, not global optimality.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def verify_predictions(rows, games):
    indexed = {(r['seed'], r['answer_index'], r['turn']): r for r in rows}
    assert len(indexed) == len(rows), 'Duplicate decision records'
    checked = set()
    for game in games:
        assert 1 <= len(game['steps']) <= 6
        assert game['won'] == (game['steps'][-1]['feedback'] == 'GGGGG')
        assert game['won'] or len(game['steps']) == 6
        history = []
        for step in game['steps']:
            key = (game['seed'], game['answer_index'], step['turn'])
            assert key not in checked
            checked.add(key)
            row = indexed[key]
            probabilities = np.asarray(step['probabilities'], dtype=float)
            action = step['action']
            assert len(probabilities) == len(step['words']) == 64
            assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
            assert np.isclose(probabilities.sum(), 1, atol=1e-6, rtol=0)
            assert action == int(probabilities.argmax())
            assert row['confidence'] == float(probabilities[action])
            assert row['chosen_word'] == step['chosen_word'] == step['words'][action]
            assert step['history'] == history
            assert row['chosen_q'] <= row['best_q'] + 1e-10
            solver_best = row['chosen_q'] >= row['best_q'] - 1e-10
            assert row['solver_best'] == int(solver_best)
            if row['closing_opportunity']:
                assert row['remaining_count'] == 1
                assert row['closing_correct'] == int(row['chosen_consistent'])
            expected = row['closing_correct'] if row['closing_opportunity'] else solver_best
            assert row['closing_aware_best'] == int(expected)
            history.append([step['chosen_word'], step['feedback']])
    assert checked == set(indexed), 'Games and decision records differ'


def normalize_records(rows, games):
    """Accept both the original SFT archive and current complete-game evaluator."""
    rows = [dict(r) for r in rows]
    for row in rows:
        row.setdefault('remaining_count', row.get('n'))
        row.setdefault('solver_best', int(row['chosen_q'] >= row['best_q'] - 1e-12))
    normalized = []
    for game in games:
        game = {**game, 'steps': [dict(s) for s in game['steps']]}
        for turn, step in enumerate(game['steps'], 1):
            step.setdefault('turn', turn)
            if 'feedback' not in step:
                step['feedback'] = game['history'][turn - 1][1]
            if 'probabilities' not in step:
                step['probabilities'] = np.exp(np.asarray(step['raw_log_probabilities'], dtype=float)).tolist()
        normalized.append(game)
    return rows, normalized


def calculate(rows, label, bins, answer_ids, weights):
    # Cluster by answer so the bootstrap keeps all turns and all ordering seeds.
    clusters = np.zeros((len(answer_ids), bins, 3), dtype=float)
    for row in rows:
        confidence = row['confidence']
        assert 0 <= confidence <= 1
        bin_index = min(bins - 1, int(confidence * bins))
        clusters[answer_ids[row['answer_index']], bin_index] += (
            1, confidence, row[label])
    totals = clusters.sum(axis=0)
    count = int(totals[:, 0].sum())
    drawn = (weights @ clusters.reshape(len(answer_ids), -1)).reshape(-1, bins, 3)
    denominators = drawn[:, :, 0].sum(axis=1)
    valid = denominators > 0
    bootstrap_ece = (np.abs(drawn[:, :, 2] - drawn[:, :, 1]).sum(axis=1)[valid]
                     / denominators[valid])
    bin_intervals = []
    for i in range(bins):
        nonempty = drawn[:, i, 0] > 0
        accuracies = drawn[nonempty, i, 2] / drawn[nonempty, i, 0]
        bin_intervals.append({
            'best_choice_accuracy_95ci': np.quantile(accuracies, [0.025, 0.975]).tolist()
            if nonempty.any() else None,
            'bootstrap_nonempty_draws': int(nonempty.sum()),
        })
    return {
        'decisions': count,
        'best_choice_accuracy': float(totals[:, 2].sum() / count) if count else None,
        'mean_top_probability': float(totals[:, 1].sum() / count) if count else None,
        'ece': float(np.abs(totals[:, 2] - totals[:, 1]).sum() / count) if count else None,
        'ece_95ci': np.quantile(bootstrap_ece, [0.025, 0.975]).tolist() if valid.any() else None,
        'reliability_bins': [
            {'low': i / bins, 'high': (i + 1) / bins, 'count': int(n),
             'mean_top_probability': float(p / n) if n else None,
             'best_choice_accuracy': float(y / n) if n else None,
             **bin_intervals[i]}
            for i, (n, p, y) in enumerate(totals)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--bins', type=int, default=15)
    parser.add_argument('--bootstrap-draws', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=722)
    args = parser.parse_args()
    if args.bins < 1 or args.bootstrap_draws < 1:
        parser.error('Bins and bootstrap draws must be positive')
    sources = {}

    def read(name):
        data = (args.evaluation / name).read_bytes()
        sources[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    protocol = read('protocol.json')
    row_file = 'decision_analysis.json' if (args.evaluation/'decision_analysis.json').exists() else 'decision_metrics.json'
    rows = read(row_file)
    games = [g for seed in protocol['seeds'] for g in read(f'games_seed{seed}.json')]
    rows, games = normalize_records(rows, games)
    verify_predictions(rows, games)
    answer_ids = {answer: i for i, answer in enumerate(sorted({r['answer_index'] for r in rows}))}
    n_answers = len(answer_ids)
    assert n_answers == protocol['unique_answers']
    assert len(games) == n_answers * len(protocol['seeds'])
    draws = np.random.default_rng(args.seed).integers(
        0, n_answers, size=(args.bootstrap_draws, n_answers))
    weights = np.asarray([np.bincount(draw, minlength=n_answers) for draw in draws], dtype=float)
    subsets = {'all': rows}
    subsets.update({f'turn_{t}': [r for r in rows if r['turn'] == t] for t in range(1, 7)})
    subsets['turn_5_remaining_gt_2'] = [r for r in rows if r['turn'] == 5 and r['remaining_count'] > 2]
    subsets['public_singleton_offered'] = [r for r in rows if r['closing_opportunity']]
    results = {label: {name: calculate(group, label, args.bins, answer_ids, weights)
                       for name, group in subsets.items()}
               for label in ['solver_best', 'closing_aware_best']}
    report = {
        'run': protocol.get('run', protocol.get('checkpoint') or protocol.get('policy')),
        'checkpoint_step': protocol.get('checkpoint_step'),
        'checkpoint_policy_sha256': protocol.get('expected_policy_sha256'),
        'split': protocol['split'], 'unique_answers': n_answers,
        'games': len(games), 'decisions': len(rows), 'ordering_seeds': protocol['seeds'],
        'confidence': 'Original greedy-action probability; no temperature calibration fitted.',
        'best_definitions': {
            'solver_best': 'Maximal public solver continuation win value among offered actions; all ties accepted (tolerance 1e-10).',
            'closing_aware_best': 'Same solver criterion, except require the unique publicly consistent answer immediately when it is offered.',
        },
        'solver_scope': 'Turns 5-6 exact optimal value under the public uniform-answer posterior; turns 1-4 exact value for the fixed solver continuation, not global optimality.',
        'ece_definition': 'Sum over bins of (count / all decisions) * abs(mean top probability - best-choice accuracy).',
        'bins': args.bins, 'binning': 'Equal width on [0,1]; left closed, right open except final bin includes 1.',
        'bootstrap': {'unit': 'answer cluster, retaining all seeds and turns',
                      'draws': args.bootstrap_draws, 'seed': args.seed,
                      'bin_intervals': 'Pointwise percentile intervals for bin accuracy, omitting bootstrap draws with an empty bin; not simultaneous bands.'},
        'verification': 'Every saved decision matched to complete-game probabilities and actions; binary labels recomputed from archived solver values and closing flags. Solver search not rerun.',
        'source_sha256': sources,
        'metrics': results,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix('.json')
    csv_path = args.output_prefix.with_suffix('.csv')
    json_path.write_text(json.dumps(report, indent=2) + '\n')
    flat = []
    for label, groups in results.items():
        for name, result in groups.items():
            low, high = result['ece_95ci'] or (None, None)
            flat.append({'label': label, 'subset': name,
                         **{k: result[k] for k in ['decisions', 'mean_top_probability', 'best_choice_accuracy', 'ece']},
                         'ece_95ci_low': low, 'ece_95ci_high': high})
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(flat)
    print(f'Verified {len(rows)} decisions from {len(games)} complete games.')
    print(csv_path.read_text())
    print(f'Reliability bins and provenance: {json_path}')


if __name__ == '__main__':
    main()
