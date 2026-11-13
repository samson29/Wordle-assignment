"""Plot held-out victory-turn histograms and export their exact values.

Run from the repository root: python scripts/plot_test_turns.py
Means are recalculated from the win histograms. Confidence intervals come from
the saved 2,000-replicate answer-cluster bootstrap (500 answers, seed 722), which
retains all three option-order seeds for each answer. No GPU/model is needed.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ('PSL-head-consloss@1100', 'PSL-head-consloss @1100', 'Starting checkpoint', '#567c91'),
    ('RL-A@12', 'RL-A @12', 'Curriculum + probe loss', '#218b7b'),
    ('RL-B@12', 'RL-B @12', 'Additional low-information penalty', '#cb8335'),
]
OUTCOMES = [str(i) for i in range(1, 7)] + ['failed']


def extract(document):
    models, histogram_rows, summary_rows = [], [], []
    for key, label, subtitle, color in MODELS:
        source = document[key]
        source = source.get('performance', source)
        total, wins = source['games'], source['wins']
        assert total == 1500 and source['unique_answers'] == 500
        counts = []
        for outcome in OUTCOMES:
            entry = source['guess_distribution'][outcome]
            count = int(entry.get('count', entry.get('numerator')))
            fraction = entry.get('fraction', entry.get('mean'))
            low, high = entry['answer_bootstrap_95pct']
            assert np.isclose(count / total, fraction)
            counts.append(count)
            histogram_rows.append({'model': label, 'outcome': outcome, 'count': count,
                'fraction_of_all_games': fraction, 'fraction_ci_low': low,
                'fraction_ci_high': high, 'count_ci_low': low * total,
                'count_ci_high': high * total})
        assert sum(counts) == total and sum(counts[:6]) == wins
        mean = float(np.dot(np.arange(1, 7), counts[:6]) / wins)
        saved = source['mean_guesses_wins']
        if isinstance(saved, dict):
            saved_mean, interval = saved['mean'], saved['answer_bootstrap_95pct']
        else:
            saved_mean, interval = saved, source['mean_guesses_wins_95pct']
        assert np.isclose(mean, saved_mean, atol=1e-14, rtol=0)
        assert 1 <= interval[0] <= mean <= interval[1] <= 6
        summary_rows.append({'model': label, 'games': total, 'wins': wins,
            'losses': total - wins, 'win_rate': wins / total,
            'mean_guesses_wins': mean, 'mean_guesses_wins_ci_low': interval[0],
            'mean_guesses_wins_ci_high': interval[1], 'ci_level': .95,
            'bootstrap_unit': 'answer', 'bootstrap_replicates': 2000, 'bootstrap_seed': 722})
        models.append({'label': label, 'subtitle': subtitle, 'color': color,
            'counts': counts, 'total': total, 'wins': wins, 'mean': mean, 'ci': interval})
    return models, histogram_rows, summary_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def plot(models, prefix):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#9aa2a8', 'axes.linewidth': .7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.8), sharey=True)
    fig.subplots_adjust(left=.065, right=.985, bottom=.31, top=.73, wspace=.13)
    fig.suptitle('Held-out games: turn of victory', x=.525, y=.975,
                 fontsize=17, fontweight='bold')
    fig.text(.525, .91, '500 held-out answers × 3 option-order seeds = 1,500 games per model',
             ha='center', fontsize=11, color='#49535a')
    positions = np.arange(7)
    for ax, model in zip(axes, models):
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e8ecef', linewidth=.7)
        ax.bar(positions[:6], model['counts'][:6], width=.72,
               color=model['color'], edgecolor='white', linewidth=.5)
        ax.bar(positions[6], model['counts'][6], width=.72,
               facecolor='#e0e4e7', edgecolor='#727c84', hatch='///', linewidth=.7)
        ax.axvline(5.5, color='#bfc6cb', linestyle=(0, (3, 3)), linewidth=.8)
        for x, count in zip(positions, model['counts']):
            ax.text(x, count + 9, str(count), ha='center', va='bottom',
                    fontsize=10, fontweight='normal', color='#28343b')
        ax.set_ylim(0, 470)
        ax.set_xlim(-.62, 6.65)
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.set_xticks(positions, ['1', '2', '3', '4', '5', '6', 'Failed'])
        ax.tick_params(axis='both', labelsize=10, length=3)
        ax.set_xlabel('Winning turn / failure', labelpad=9)
        ax.set_title(model['label'], fontsize=12, fontweight='bold', color=model['color'], pad=44)
        ax.text(.5, 1.115, model['subtitle'], transform=ax.transAxes,
                ha='center', fontsize=9.4, color='#576269')
        ax.text(.5, 1.035, f"{model['wins']:,} / {model['total']:,} wins ({model['wins']/model['total']:.2%})",
                transform=ax.transAxes, ha='center', fontsize=10.2)
        box = ax.get_position()
        center = (box.x0 + box.x1) / 2
        fig.text(center, .18, f"Mean guesses among wins: {model['mean']:.3f}",
                 ha='center', fontsize=11, fontweight='bold', color='#28343b')
        fig.text(center, .135, f"95% CI [{model['ci'][0]:.3f}, {model['ci'][1]:.3f}]",
                 ha='center', fontsize=10.5, color='#49535a')
    axes[0].set_ylabel('Number of games', labelpad=9)
    fig.text(.065, .065, 'Bars show observed counts. Failed games are shown separately and excluded from the mean.',
             fontsize=9, color='#49535a')
    fig.text(.065, .027, '95% CIs: 2,000 answer-cluster bootstrap samples, preserving all three ordering seeds for each answer.',
             fontsize=9, color='#49535a')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix('.pdf'), bbox_inches='tight', pad_inches=.12,
                metadata={'Title': 'LetterLock held-out victory-turn histograms',
                          'Creator': 'LetterLock scripts/plot_test_turns.py',
                          'CreationDate': None, 'ModDate': None})
    fig.savefig(prefix.with_suffix('.png'), dpi=240, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, default=ROOT/'results/heldout_metrics.json')
    parser.add_argument('--out', type=Path, default=ROOT/'report/figures/test_victory_turns')
    parser.add_argument('--results', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    models, bins, summary = extract(json.loads(args.metrics.read_text()))
    write_csv(args.results/'test_victory_turns.csv', bins)
    write_csv(args.results/'test_victory_turns_summary.csv', summary)
    plot(models, args.out)
    print(json.dumps({'summary': summary, 'figure': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
