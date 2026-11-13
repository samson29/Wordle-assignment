"""Three overall reliability plots for the starting policy and both RL policies.

Compute and plot from the experiment archive (CPU only):
    python scripts/plot_overall_calibration.py --archive ../FINAL/heldout
Redraw from the bundled result without access to the archive:
    python scripts/plot_overall_calibration.py
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
from calibration import calculate

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ('PSL-head-consloss@1100', 'PSL-head-consloss @1100',
     'step1100/PSL-head-consloss', 'psl_head_consloss_1100', '#567c91'),
    ('RL-A@12', 'RL-A @12', 'rl_iteration12/D', 'rl_a_12', '#218b7b'),
    ('RL-B@12', 'RL-B @12', 'rl_iteration12/E', 'rl_b_12', '#cb8335'),
]


def read_json(path, source_hashes, archive):
    data = path.read_bytes()
    source_hashes[str(path.relative_to(archive))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def verify(rows, games):
    indexed = {(r['seed'], r['answer_index'], r['turn']): r for r in rows}
    assert len(indexed) == len(rows)
    checked, game_keys = set(), set()
    for game in games:
        game_key = (game['seed'], game['answer_index'])
        assert game_key not in game_keys
        game_keys.add(game_key)
        history = []
        for turn, step in enumerate(game['steps'], 1):
            key = (*game_key, turn)
            assert key not in checked
            checked.add(key)
            row = indexed[key]
            if 'raw_log_probabilities' in step:
                probabilities = np.exp(np.asarray(step['raw_log_probabilities'], dtype=np.float64))
                feedback = game['history'][turn - 1][1]
            else:
                probabilities = np.asarray(step['probabilities'], dtype=np.float64)
                feedback = step['feedback']
            action = step['action']
            assert probabilities.shape == (64,) and np.isfinite(probabilities).all()
            assert np.isclose(probabilities.sum(), 1, atol=1e-5, rtol=0)
            assert action == int(probabilities.argmax())
            assert row['chosen_word'] == step['chosen_word'] == step['words'][action]
            assert np.isclose(row['confidence'], probabilities[action], atol=1e-14, rtol=0)
            assert step['history'] == history
            assert row['chosen_q'] <= row['best_q'] + 1e-10
            correct = int(row['chosen_q'] >= row['best_q'] - 1e-12)
            if row['closing_opportunity']:
                assert row.get('remaining_count', row.get('n')) == 1
                assert row['closing_correct'] == int(row['chosen_consistent'])
                correct = row['closing_correct']
            assert row['closing_aware_best'] == correct
            history.append([step['chosen_word'], feedback])
        assert 1 <= len(history) <= 6
        assert game['won'] == (history[-1][1] == 'GGGGG')
        assert game['won'] or len(history) == 6
    assert game_keys == {(seed, i) for seed in [1001, 1002, 1003] for i in range(500)}
    assert checked == set(indexed)


def compute(archive):
    archive = archive.resolve()
    hashes, results = {}, []
    draws = np.random.default_rng(722).integers(0, 500, size=(2000, 500))
    weights = np.asarray([np.bincount(d, minlength=500) for d in draws], dtype=float)
    expected = json.loads((ROOT/'results/heldout_metrics.json').read_text())
    for key, label, folder, stem, color in MODELS:
        source = archive / folder
        row_file = source / ('decision_analysis.json' if key.startswith('PSL') else 'decision_metrics.json')
        rows = read_json(row_file, hashes, archive)
        paths = sorted(source.glob('games_seed*.json')) if key.startswith('PSL') else sorted(source.glob('shards/*/games_seed*.json'))
        assert len(paths) == 3
        games = [game for path in paths for game in read_json(path, hashes, archive)]
        verify(rows, games)
        metric = calculate(rows, 'closing_aware_best', 15, {i: i for i in range(500)}, weights)
        saved = expected[key]
        saved = saved['endgame_and_calibration']['calibration_closing_aware'] if key.startswith('PSL') else saved['calibration']
        assert metric['decisions'] == saved['decisions']
        np.testing.assert_allclose([metric['ece'], *metric['ece_95ci']],
            [saved['ECE'], *saved['answer_bootstrap_95pct']], atol=1e-14, rtol=0)
        results.append({'model': label, 'stem': stem, 'color': color, **metric})
    return {
        'population': 'All observed decisions from each model\'s own 1,500 complete held-out games; no turn filtering or deduplication.',
        'unique_answers': 500, 'games_per_model': 1500, 'ordering_seeds': [1001, 1002, 1003],
        'confidence': 'Raw probability of the greedy chosen option; no temperature calibration.',
        'correct_choice': 'Chosen option has maximal public solver continuation win value; all ties count. In public-singleton states, require the known answer immediately.',
        'solver_scope': 'Exact two-turn/one-turn values on turns 5/6; fixed solver continuation on earlier turns, not global optimality.',
        'ece_definition': 'Sum over bins of count/N * abs(mean confidence - best-choice accuracy).',
        'bins': 15, 'binning': 'Equal-width bins on [0,1]; final bin includes 1; empty bins omitted from curves.',
        'bootstrap': {'unit': 'answer cluster retaining all turns and all three seeds', 'draws': 2000, 'seed': 722, 'confidence': .95,
                      'bin_intervals': 'Pointwise percentile CIs, excluding bootstrap samples where the bin is empty.'},
        'verification': 'All decisions matched to saved complete-game greedy actions and raw probabilities. Correct-choice labels checked against saved solver values and closing flags; all ECE values and CIs match previous evaluation outputs.',
        'sources_sha256': hashes, 'models': results,
    }


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def export(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    summary, bins = [], []
    for model in report['models']:
        summary.append({'model': model['model'], 'decisions': model['decisions'],
            'mean_top_probability': model['mean_top_probability'], 'best_choice_accuracy': model['best_choice_accuracy'],
            'ece': model['ece'], 'ece_pct': 100 * model['ece'],
            'ece_95ci_low': model['ece_95ci'][0], 'ece_95ci_high': model['ece_95ci'][1]})
        for row in model['reliability_bins']:
            low, high = row['best_choice_accuracy_95ci'] or (None, None)
            bins.append({'model': model['model'], **{k: row[k] for k in
                ['low', 'high', 'count', 'mean_top_probability', 'best_choice_accuracy']},
                'accuracy_ci_low': low, 'accuracy_ci_high': high,
                'bootstrap_nonempty_draws': row['bootstrap_nonempty_draws']})
    write_csv(path.with_suffix('.csv'), summary)
    write_csv(path.with_name(path.stem + '_bins.csv'), bins)
    print(json.dumps(summary, indent=2))


def panel(ax, model, max_count, y_label=True):
    color = model['color']
    rows = [r for r in model['reliability_bins'] if r['count']]
    x = np.asarray([r['mean_top_probability'] for r in rows])
    y = np.asarray([r['best_choice_accuracy'] for r in rows])
    ci = np.asarray([r['best_choice_accuracy_95ci'] for r in rows])
    counts = np.asarray([r['count'] for r in rows])
    ax.plot([0, 1], [0, 1], color='#657078', linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
    ax.vlines(x, ci[:, 0], ci[:, 1], color=color, linewidth=1, alpha=.48, zorder=2)
    ax.hlines(ci[:, 0], x-.008, x+.008, color=color, linewidth=1, alpha=.48)
    ax.hlines(ci[:, 1], x-.008, x+.008, color=color, linewidth=1, alpha=.48)
    ax.plot(x, y, color=color, linewidth=1.35, zorder=3)
    ax.scatter(x, y, s=18 + 70*np.sqrt(counts/max_count), color=color,
               edgecolors='white', linewidths=.55, zorder=4)
    ax.set_xlim(-.015, 1.025); ax.set_ylim(-.015, 1.025)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.linspace(0, 1, 6)); ax.set_yticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(color='#e9edef', linewidth=.65)
    ax.set_axisbelow(True)
    ax.set_xlabel('Top-option probability', labelpad=8)
    if y_label: ax.set_ylabel('Best-choice accuracy', labelpad=8)
    else: ax.tick_params(axis='y', labelleft=False)
    ax.set_title(f"{model['model']}\n{model['decisions']:,} decisions", color=color,
                 fontsize=11.5, fontweight='bold', linespacing=1.7, pad=15)


def plot(report, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8c979e',
        'axes.linewidth': .7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})
    output.mkdir(parents=True, exist_ok=True)
    models = report['models']
    max_count = max(r['count'] for m in models for r in m['reliability_bins'])
    meta = {'Title': 'Overall held-out best-choice calibration',
            'Creator': 'LetterLock scripts/plot_overall_calibration.py', 'CreationDate': None, 'ModDate': None}
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 6.7))
    fig.subplots_adjust(left=.065, right=.985, bottom=.30, top=.77, wspace=.16)
    fig.suptitle('Overall calibration · all played turns', x=.525, y=.975, fontsize=16, fontweight='bold')
    fig.text(.525, .918, '500 held-out answers × 3 option-order seeds · 1,500 complete games per model',
             ha='center', fontsize=10.5, color='#49535a')
    for i, (ax, model) in enumerate(zip(axes, models)):
        panel(ax, model, max_count, i == 0)
        position = ax.get_position(); center = (position.x0 + position.x1)/2
        lo, hi = model['ece_95ci']
        fig.text(center, .185, f"ECE: {100*model['ece']:.2f}%  [95% CI {100*lo:.2f}–{100*hi:.2f}%]",
                 ha='center', fontsize=10, fontweight='bold', color=model['color'])
    fig.legend(handles=[
        Line2D([0], [0], color='#657078', linestyle=(0, (4, 3)), label='Perfect calibration'),
        Line2D([0], [0], color='#435159', marker='o', markersize=5, label='Observed bin accuracy'),
        Line2D([0], [0], color='#657078', marker='|', markersize=12, linestyle='none', label='95% pointwise bin CI'),
    ], loc='lower center', bbox_to_anchor=(.525, .12), ncol=3, frameon=False, fontsize=9)
    fig.text(.065, .092, '15 equal-width bins; raw probabilities; larger markers indicate more decisions. CIs: 2,000 answer-cluster bootstrap samples.', fontsize=8.5)
    fig.text(.065, .061, 'Best = maximal public-solver win value among offered words, requiring immediate closing when the answer is uniquely known.', fontsize=8.5)
    fig.text(.065, .030, 'Earlier turns use fixed-continuation solver values; turns 5–6 are exact. Each model is measured on its own played states.', fontsize=8.5)
    prefix = output / 'overall_calibration_comparison'
    fig.savefig(prefix.with_suffix('.pdf'), bbox_inches='tight', pad_inches=.12, metadata=meta)
    fig.savefig(prefix.with_suffix('.png'), dpi=240, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    # Separate vector files let the report place the same three plots individually.
    for model in models:
        fig, ax = plt.subplots(figsize=(4.5, 5.7))
        fig.subplots_adjust(left=.16, right=.96, bottom=.32, top=.79)
        panel(ax, model, max_count)
        lo, hi = model['ece_95ci']
        fig.text(.56, .16, f"ECE {100*model['ece']:.2f}% · 95% CI [{100*lo:.2f}, {100*hi:.2f}]%", ha='center', fontsize=10)
        fig.text(.08, .075, 'All turns · 15 equal-width bins · raw probabilities', fontsize=8.5)
        fig.text(.08, .039, '95% bin CIs: answer-cluster bootstrap (2,000 samples)', fontsize=8.5)
        fig.savefig(output/f"overall_calibration_{model['stem']}.pdf", bbox_inches='tight', pad_inches=.12, metadata=meta)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='FINAL/heldout directory; omit to redraw bundled data')
    parser.add_argument('--results', type=Path, default=ROOT/'results/overall_calibration_comparison.json')
    parser.add_argument('--out', type=Path, default=ROOT/'report/figures')
    args = parser.parse_args()
    report = compute(args.archive) if args.archive else json.loads(args.results.read_text())
    export(report, args.results)
    plot(report, args.out)


if __name__ == '__main__':
    main()
