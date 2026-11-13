"""Create matched PDF/PNG reliability diagrams from calibration.py JSON outputs.

Example, from the repository root:
    python scripts/plot_calibration.py results/calibration_psl_head*_1100.json

No model loading or GPU use. Each checkpoint is evaluated on its own reached
states; the diagrams do not imply a comparison on identical decision states.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter
import numpy as np


PANELS = [
    ('all', '(a) All turns', '#25658a'),
    ('turn_5', '(b) Turn 5', '#b85c22'),
    ('turn_5_remaining_gt_2', '(c) Turn 5: >2 answers left', '#76558e'),
]


def plot(source, output_dir):
    report = json.loads(source.read_text())
    metrics = report['metrics']['closing_aware_best']
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 10.5, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#7b8186', 'axes.linewidth': 0.7,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white',
    })
    fig = plt.figure(figsize=(9.4, 6.2))
    grid = fig.add_gridspec(2, 3, height_ratios=[3.1, 1], hspace=0.14,
                            wspace=0.22, left=0.078, right=0.985,
                            bottom=0.285, top=0.825)
    title = f"{report['run']} · step {report['checkpoint_step']}"
    fig.suptitle(title, x=0.53, y=0.982, fontsize=14, fontweight='bold')
    fig.text(0.53, 0.941,
             f"{report['unique_answers']:,} held-out answers × {len(report['ordering_seeds'])} option-order seeds · "
             f"{report['games']:,} complete games", ha='center', fontsize=10, color='#444b50')

    for col, (key, name, color) in enumerate(PANELS):
        result = metrics[key]
        bins = result['reliability_bins']
        assert len(bins) == report['bins']
        assert sum(b['count'] for b in bins) == result['decisions']
        populated = [b for b in bins if b['count']]
        x = np.asarray([b['mean_top_probability'] for b in populated])
        y = np.asarray([b['best_choice_accuracy'] for b in populated])
        ci = np.asarray([b['best_choice_accuracy_95ci'] for b in populated])
        assert ci.shape == (len(populated), 2) and np.isfinite(ci).all()
        assert (ci[:, 0] >= 0).all() and (ci[:, 1] <= 1).all()
        assert np.all(ci[:, 0] <= ci[:, 1])

        ax = fig.add_subplot(grid[0, col])
        ax.set_title(f"{name}\nN = {result['decisions']:,}  |  ECE = {100 * result['ece']:.2f} pp",
                     loc='left', pad=10, linespacing=1.5)
        ax.plot([0, 1], [0, 1], color='#737a80', linestyle=(0, (4, 3)),
                linewidth=1.1, zorder=1)
        # Draw percentile endpoints directly: a bootstrap interval need not
        # contain its point estimate, unlike matplotlib's errorbar convention.
        ax.vlines(x, ci[:, 0], ci[:, 1], color=color, alpha=0.4, linewidth=1.1, zorder=2)
        ax.hlines(ci[:, 0], x - 0.009, x + 0.009, color=color, alpha=0.4, linewidth=1.1)
        ax.hlines(ci[:, 1], x - 0.009, x + 0.009, color=color, alpha=0.4, linewidth=1.1)
        ax.plot(x, y, '-o', color=color, markersize=4.2, linewidth=1.35,
                markeredgecolor='white', markeredgewidth=0.5, zorder=3)
        ax.set_xlim(-0.015, 1.02)
        ax.set_ylim(-0.015, 1.025)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(axis='x', labelbottom=False)
        ax.grid(axis='both', color='#e9ecee', linewidth=0.6, zorder=0)
        if col == 0:
            ax.set_ylabel('Best-choice accuracy')
        else:
            ax.tick_params(axis='y', labelleft=False)

        hist = fig.add_subplot(grid[1, col], sharex=ax)
        left = np.asarray([b['low'] for b in bins])
        width = np.asarray([b['high'] - b['low'] for b in bins])
        counts = np.asarray([b['count'] for b in bins])
        shares = 100 * counts / result['decisions']
        present = counts > 0
        # Empty bins are absent, not drawn as zero-accuracy predictions.
        hist.bar(left[present] + 0.04 * width[present], shares[present],
                 width=0.92 * width[present], align='edge', color=color, alpha=0.75,
                 linewidth=0, zorder=2)
        hist.set_yscale('log')
        hist.set_ylim(0.1, 120)
        hist.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
        hist.yaxis.set_minor_locator(NullLocator())
        hist.yaxis.set_major_formatter(PercentFormatter(100, decimals=1))
        hist.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        hist.set_xlabel('Top-option probability', labelpad=5)
        hist.grid(axis='y', color='#e9ecee', linewidth=0.6, zorder=0)
        if col == 0:
            hist.set_ylabel('Decisions\n(log scale)', fontsize=9)
        else:
            hist.tick_params(axis='y', labelleft=False)

    fig.legend(handles=[
        Line2D([0], [0], color='#737a80', linestyle=(0, (4, 3)), label='Perfect calibration'),
        Line2D([0], [0], color='#444b50', marker='o', markersize=4, label='Observed bin accuracy'),
        Line2D([0], [0], color='#777d82', marker='|', markersize=12,
               linestyle='none', label='95% pointwise bootstrap CI'),
    ], loc='lower center', bbox_to_anchor=(0.53, 0.158), ncol=3,
       frameon=False, fontsize=9, handlelength=2.2, columnspacing=1.5)
    fig.text(0.078, 0.126,
             f"{report['bins']} equal-width bins; raw probabilities. CIs resample {report['unique_answers']} answer clusters "
             f"({report['bootstrap']['draws']:,} draws), preserving all turns and seeds.", fontsize=8.3)
    fig.text(0.078, 0.098,
             'Best = maximal solver value (ties accepted), with immediate closing required when the answer is uniquely known.',
             fontsize=8.3)
    fig.text(0.078, 0.070,
             'Solver values: fixed continuation on turns 1–4; exact endgame values on turns 5–6. Below the diagonal = overconfidence.',
             fontsize=8.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / source.stem.replace('calibration_', 'reliability_', 1)
    metadata = {'Title': title + ' — reliability diagram',
                'Subject': 'Closing-aware public-solver calibration on held-out complete games',
                'Creator': 'LetterLock scripts/plot_calibration.py',
                'CreationDate': None, 'ModDate': None}
    fig.savefig(prefix.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.12, metadata=metadata)
    fig.savefig(prefix.with_suffix('.png'), dpi=240, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)
    print(prefix.with_suffix('.pdf'))
    print(prefix.with_suffix('.png'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('calibration', type=Path, nargs='+')
    parser.add_argument('--out', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'report' / 'figures')
    args = parser.parse_args()
    for source in args.calibration:
        plot(source, args.out)


if __name__ == '__main__':
    main()
