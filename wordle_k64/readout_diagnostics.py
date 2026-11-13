"""Fixed validation-state panel, independent of training RNG and no test secrets."""
import json
from pathlib import Path
import numpy as np
import torch
from .environment import Decision


def load_panel(path, per_turn=12):
    unique = {}
    for line in Path(path).read_text().splitlines():
        for s in json.loads(line)['steps']:
            d = Decision.from_dict(s)
            unique.setdefault(d.history, s)
    rng = np.random.default_rng(20261116)
    panel = []
    for turn in range(1, 7):
        rows = [s for h, s in unique.items() if len(h) == turn-1]
        order = rng.permutation(len(rows))[:per_turn]
        panel.extend(rows[i] for i in order)
    return panel


def evaluate_order(policy, decisions, batch_size=8):
    """Same batches, word-aligned comparisons; fixed reversal plus two random orders."""
    original = policy.score(decisions, batch_size).exp()
    repeat = policy.score(decisions, batch_size).exp()
    same_flips = (original.argmax(-1) != repeat.argmax(-1)).float()
    same_tv = (original-repeat).abs().sum(-1)*0.5
    result = {'states': len(decisions), 'permutations_per_state': 3,
              'comparisons': 3 * len(decisions), 'batch_size': batch_size,
              'permutation_seed': 20261117,
              'permutation_scheme': ['reverse', 'uniform_random_1', 'uniform_random_2'],
              'same_order_flip_rate': float((original.argmax(-1) != repeat.argmax(-1)).float().mean()),
              'same_order_mean_total_variation': float(same_tv.mean()),
              'same_order_max_probability_delta': float((original-repeat).abs().max()),
              'by_turn': {}}
    rng = np.random.default_rng(20261117)
    flips, tvs, orders, aligned = [], [], [], []
    for variant in range(3):
        permutations = [np.arange(63, -1, -1) if variant == 0 else rng.permutation(64) for _ in decisions]
        shuffled = [Decision(d.history, tuple(d.words[i] for i in perm)) for d, perm in zip(decisions, permutations)]
        changed = policy.score(shuffled, batch_size).exp()
        mapped = torch.stack([row[torch.as_tensor(perm.argsort().copy())] for row, perm in zip(changed, permutations)])
        flips.append((original.argmax(-1) != mapped.argmax(-1)).float())
        tvs.append((original-mapped).abs().sum(-1)*0.5)
        orders.append(permutations)
        aligned.append(mapped)
    result['order_flip_rate'] = float(torch.stack(flips).mean())
    result['mean_total_variation'] = float(torch.stack(tvs).mean())
    result['flipped_comparisons'] = int(torch.stack(flips).sum())
    result['states_with_any_flip'] = int(torch.stack(flips).bool().any(0).sum())
    result['any_order_flip_rate'] = result['states_with_any_flip'] / len(decisions)
    for turn in range(1, 7):
        ids = [i for i, d in enumerate(decisions) if len(d.history) == turn-1]
        if ids:
            result['by_turn'][str(turn)] = {'states': len(ids), 'comparisons': 3*len(ids),
                'order_flip_rate': float(torch.stack(flips)[:, ids].mean()),
                'mean_total_variation': float(torch.stack(tvs)[:, ids].mean()),
                'same_order_flip_rate': float(same_flips[ids].mean()),
                'same_order_mean_total_variation': float(same_tv[ids].mean())}
    rows = [{**d.to_dict(), 'probabilities': original[i].tolist(),
             'repeat_probabilities': repeat[i].tolist(),
             'permutations': [order[i].tolist() for order in orders],
             'aligned_permuted_probabilities': [p[i].tolist() for p in aligned]}
            for i, d in enumerate(decisions)]
    return result, original, rows


def evaluate_panel(policy, panel, batch_size=8, include_order_rows=False):
    decisions = [Decision.from_dict(s) for s in panel]
    result, original, rows = evaluate_order(policy, decisions, batch_size)
    for turn in range(1, 7):
        indices = [i for i, d in enumerate(decisions) if len(d.history) == turn-1]
        if not indices: continue
        masses = [float(original[i, panel[i]['accepted']].sum()) for i in indices]
        closing = [i for i in indices if panel[i].get('closing_target') is not None]
        row = {'decisions': len(indices), 'accepted_probability': float(np.mean(masses)),
               'accepted_loss': float(np.mean(-np.log(np.maximum(masses, 1e-30)))),
               'accepted_accuracy': float(np.mean([int(original[i].argmax()) in panel[i]['accepted'] for i in indices])),
               'closing_decisions': len(closing)}
        if closing:
            row['closing_accuracy'] = float(np.mean([int(original[i].argmax()) == panel[i]['closing_target'] for i in closing]))
            row['closing_probability'] = float(np.mean([float(original[i, panel[i]['closing_target']]) for i in closing]))
        result['by_turn'][str(turn)].update(row)
    if include_order_rows:
        result['order_rows'] = rows
    return result
