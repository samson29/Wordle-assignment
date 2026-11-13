"""Snapshot completed metrics from the experiment archive; never modify its files or jobs."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from _common import ROOT, RUN_TIMESTAMP_KEYS


SFT = {'PSL-head': 'PSL-head', 'PSL-head-consloss': 'PSL-head-consloss',
       'LM-head': 'lm-head-readout'}
RL = {'RL-A': 'cons1100_probe_study/D_curriculum_grpo_probe',
      'RL-B': 'cons1100_lowinfo_penalty/E_curriculum_grpo_probe_lowinfo'}
RL_ITERATIONS = 12
CHECKPOINTS = {'PSL-head': 'step_01100', 'PSL-head-consloss': 'step_01100',
               'LM-head': 'step_01100', 'RL-A': 'iteration_00012', 'RL-B': 'iteration_00012'}
DROP = {'memory', 'decisions', 'per_answer_success', 'source_hashes', 'probabilities',
        'log_probabilities', 'order_rows', 'loading_info'}
RUN_TIMESTAMPS = RUN_TIMESTAMP_KEYS


def included_event(kind, event):
    if kind != 'rl':
        return True
    return (max(event.get('iteration', 0), event.get('next_iteration', 0)) <= RL_ITERATIONS
            and event.get('step', 0) <= RL_ITERATIONS * 8)


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        if key in RUN_TIMESTAMPS:
            continue
        name = f'{prefix}.{key}' if prefix else key
        if key in DROP and isinstance(item, (dict, list)):
            continue
        if key == 'memory':
            continue
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, list):
            if len(item) == 2 and all(isinstance(x, (float, int)) for x in item):
                result[name + '.low'], result[name + '.high'] = item
        elif item is None or isinstance(item, (int, float, bool)):
            result[name] = item
        elif key in {'stage', 'policy_sha256', 'word'}:
            result[name] = item
    return result


def csv_text(rows, leading):
    fields = leading + sorted(set().union(*(r.keys() for r in rows)) - set(leading))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='Existing FINAL directory')
    parser.add_argument('--out', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    archive = args.archive.resolve()
    sources = {}

    def read(relative):
        data = (archive / relative).read_bytes()
        sources[str(relative)] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        return data

    def clean(value):
        if isinstance(value, dict):
            return {clean(k): clean(v) for k, v in value.items()
                    if k not in RUN_TIMESTAMPS
                    and not (k in DROP and isinstance(v, (list, dict)))}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str) and str(archive.parent) in value:
            return value.replace(str(archive.parent) + '/', '')
        return value

    training, validation, run_summary = [], [], {}
    runs = [(name, f'sft/{folder}', 'sft') for name, folder in SFT.items()]
    runs += [(name, f'rl/{folder}', 'rl') for name, folder in RL.items()]
    for name, folder, kind in runs:
        configuration = json.loads(read(f'{folder}/configuration.json'))
        data = read(f'{folder}/log.jsonl')
        events = []
        lines = data.splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1 or line.endswith(b'\n'):
                    raise
                continue  # A live writer may be halfway through its last record.
            if not included_event(kind, event):
                continue
            events.append(event)
            stage = event.get('stage')
            if stage not in {'sl', 'update', 'rl', 'rollout', 'validation'}:
                continue
            row = {'run': name, 'kind': kind, **flatten(event)}
            if kind == 'rl' and 'next_iteration' in event:
                row['completed_iterations_before_event'] = event['iteration']
                row['iteration'] = event['next_iteration']
            if stage == 'validation':
                if kind == 'sft':
                    diagnostics = event.get('diagnostics', {})
                    losses = [r.get('accepted_loss', 0) * r.get('decisions', 0)
                              for r in diagnostics.get('by_turn', {}).values()]
                    count = sum(r.get('decisions', 0) for r in diagnostics.get('by_turn', {}).values())
                    row.update(validation_loss=math.fsum(losses)/count if count else None,
                               loss_population='fixed_teacher_panel',
                               flip_population='fixed_teacher_panel',
                               flip_rate=diagnostics.get('order_flip_rate'),
                               mean_tv=diagnostics.get('mean_total_variation'))
                else:
                    row.update(validation_loss=event.get('validation_sl_diagnostic', {}).get('accepted_set_loss'),
                               loss_population='fixed_teacher_panel',
                               flip_population='all_played_validation_turns',
                               flip_rate=event['order']['flip_rate']['mean'],
                               mean_tv=event['order']['mean_tv']['mean'])
                validation.append(row)
            else:
                training.append(row)
        validations = [e for e in events if e.get('stage') == 'validation']
        updates = [e for e in events if e.get('stage') in {'sl', 'update'}]
        iterations = [e for e in events if e.get('stage') == 'rl']
        run_summary[name] = {'last_update': updates[-1].get('step') if updates else None,
                             'checkpoint': f'checkpoints/{name}/{CHECKPOINTS[name]}',
                             'source_run': folder,
                             'included_through_iteration': RL_ITERATIONS if kind == 'rl' else None,
                             'status': 'stopped',
                             'last_complete_iteration': iterations[-1].get('iteration') if iterations else None,
                             'last_validation_step': validations[-1].get('step') if validations else None,
                             'last_validation_iteration': validations[-1].get('iteration') if validations else None,
                             'validation_count': len(validations),
                             'configuration': clean({k: v for k, v in configuration.items()
                                 if k in {'stage', 'seed', 'model_spec', 'checkpoint_model_spec',
                                          'data_sha256', 'target_rule', 'games_per_step', 'micro_batch',
                                          'lr', 'head_lr', 'warmup', 'consistency_coef', 'consistency_seed',
                                          'eval_seeds', 'eval_secrets', 'eval_every', 'eval_batch',
                                          'optimizer', 'reference_kl', 'reward', 'consistency', 'probe_auxiliary',
                                          'sampling_temperature', 'games_per_secret', 'games_per_iteration',
                                          'games_per_optimizer_update', 'iterations', 'ppo_ratio_clip',
                                          'curriculum'}})}

    details = {}
    for name in ['heuristic_random_opening', 'heuristic_distinct_opening', 'heuristic_salet', 'random_policy', 'zero_shot_qwen']:
        file = 'metrics.json' if name == 'zero_shot_qwen' else 'evaluation.json'
        details[name] = clean(json.loads(read(f'baselines/{name}/{file}')))
    for name in ['PSL-head', 'PSL-head-consloss']:
        details[name + '@1100'] = clean(json.loads(read(f'heldout/step1100/{name}/metrics.json')))
    for name, source_name in [('RL-A', 'D'), ('RL-B', 'E')]:
        report = clean(json.loads(read(f'heldout/rl_iteration12/{source_name}/metrics.json')))
        report['run'] = name
        report['checkpoint'] = f'checkpoints/{name}/{CHECKPOINTS[name]}'
        report['low_information']['interpretation'] = (
            "Held-out diagnostic: RL-B's penalty rule applied to both policies. "
            "RL-A was trained with unshaped terminal reward; mean_training_reward here is a diagnostic return.")
        details[name + '@12'] = report
    starting = clean(json.loads(read('heldout/rl_iteration12/starting_checkpoint_endgame.json')))
    starting['checkpoint'] = 'checkpoints/PSL-head-consloss/step_01100'
    details['PSL-head-consloss@1100']['rl_comparable_endgame'] = starting

    failures = []
    for name, answers in [('PSL-head', ['canal']), ('PSL-head-consloss', ['level', 'waxen'])]:
        games = json.loads(read(f'heldout/step1100/{name}/failed_games_enriched.json'))
        for answer in answers:
            game = next(g for g in games if g['offline_secret'] == answer)
            failures.append({'run': name, 'checkpoint_step': 1100, **clean(game)})

    dataset = json.loads(read('sft/data/episodes/stats.json'))
    source_code = [
        'sft/code/scripts/train_sl.py',
        'sft/code/scripts/make_salet_data.py',
        'sft/code/scripts/prepare_closing_data.py',
        'sft/lm-head-readout/code/wordle_k64/model.py',
        'rl/cons1100_probe_study/code/scripts/train_probe_rl.py',
        'rl/cons1100_lowinfo_penalty/code/scripts/train_probe_rl.py',
        'baselines/zero_shot_qwen/common.py',
    ]
    for relative in source_code:
        read(relative)

    heldout = list(csv.DictReader(io.StringIO(read('baselines/comparison_with_pointer_step1100.csv').decode())))
    for name in RL:
        m = details[name + '@12']
        heldout.append({'method': name, 'category': 'trained_pointer_rl', 'checkpoint_iteration': 12,
            'unique_answers': 500, 'seeds': '1001;1002;1003', 'games': m['games'], 'wins': m['wins'],
            'win_rate': m['win_rate'], 'win_rate_pct': m['win_rate'] * 100,
            'win_rate_ci_low': m['answer_bootstrap_95pct'][0], 'win_rate_ci_high': m['answer_bootstrap_95pct'][1],
            'mean_guesses_wins': m['mean_guesses_wins']['mean']})
    order_rows = []
    for name, key in [('PSL-head', 'PSL-head@1100'), ('PSL-head-consloss', 'PSL-head-consloss@1100'),
                      ('Qwen (zero-shot)', 'zero_shot_qwen'), ('RL-A', 'RL-A@12'), ('RL-B', 'RL-B@12')]:
        m = details[key]['order']
        if name in RL:
            flip, tv = m['flip_rate'], m['mean_tv']
            same = m['same_order_flip_rate']
        else:
            flip = {'mean': m['flip_rate'], 'answer_bootstrap_95pct': m['flip_rate_95pct']}
            tv = {'mean': m['mean_TV'], 'answer_bootstrap_95pct': m['mean_TV_95pct']}
            same = {'mean': m['same_order']['flip_rate'], 'answer_bootstrap_95pct': m['same_order']['flip_rate_95pct']}
        row = {'model': name, 'decisions': m['decisions'], 'comparisons': m['comparisons']}
        for metric, value in [('flip_rate', flip), ('mean_tv', tv), ('same_order_flip_rate', same)]:
            row[metric] = value['mean']
            row[metric + '_ci_low'], row[metric + '_ci_high'] = value['answer_bootstrap_95pct']
        order_rows.append(row)
    endgame = list(csv.DictReader(io.StringIO(read('heldout/rl_iteration12/endgame_comparison.csv').decode())))
    for row in endgame:
        row['model'] = {'D @12': 'RL-A @12', 'E @12': 'RL-B @12'}.get(row['model'], row['model'])
    order_by_turn = list(csv.DictReader(io.StringIO(read('heldout/rl_iteration12/order_by_turn.csv').decode())))
    for row in order_by_turn:
        row['run'] = {'D': 'RL-A', 'E': 'RL-B'}[row['run']]
    outputs = {
        'training.csv': csv_text(training, ['run', 'kind', 'stage', 'step', 'iteration']),
        'validation.csv': csv_text(validation, ['run', 'kind', 'stage', 'step', 'iteration', 'win_rate',
                                             'validation_loss', 'loss_population', 'flip_rate', 'mean_tv', 'flip_population']),
        'heldout.csv': csv_text(heldout, ['method', 'category', 'checkpoint_step', 'checkpoint_iteration',
                                       'unique_answers', 'seeds', 'games', 'wins', 'win_rate']),
        'order_robustness.csv': csv_text(order_rows, ['model', 'decisions', 'comparisons']),
        'endgame_comparison.csv': csv_text(endgame, ['model', 'metric', 'numerator', 'denominator', 'rate', 'ci_low', 'ci_high']),
        'rl_order_by_turn.csv': csv_text(order_by_turn, ['run', 'iteration', 'turn', 'decisions', 'comparisons']),
        'heldout_metrics.json': json.dumps(details, indent=2, allow_nan=False) + '\n',
        'failure_examples.json': json.dumps(failures, indent=2, allow_nan=False) + '\n',
    }
    provenance = {'source_root': 'FINAL', 'runs': run_summary, 'sources': sources,
                  'teacher_dataset': dataset,
                  'packaging': 'Single shared implementation; machine paths and launch wrappers removed. CPU mechanics tested; no new full-size training results attributed to this refactor.',
                  'notes': ['RL-A is archived run D; RL-B is archived run E. Both histories end after iteration-12 validation.',
                            'RL was originally planned for 24 iterations; this package includes iterations 1–12 and recipes ending at 12.',
                            'SFT histories retain all completed updates; the checkpoint column identifies the selected saved weights.',
                            'Wall-clock training timestamps and TensorBoard event files are omitted; elapsed durations and RNG seeds are retained.',
                            'Rates and TV are fractions, unless a column explicitly ends in _pct.',
                            'Selected checkpoints are included locally under checkpoints/. Source hashes refer to the original archive.'],
                  'exports': {name: {'sha256': hashlib.sha256(text.encode()).hexdigest(),
                                     'bytes': len(text.encode())} for name, text in outputs.items()}}
    existing = args.out/'provenance.json'
    if existing.exists():
        previous = json.loads(existing.read_text())
        provenance = {**previous, **provenance,
                      'exports': {**previous.get('exports', {}), **provenance['exports']}}
    outputs['provenance.json'] = json.dumps(provenance, indent=2) + '\n'
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        temp = args.out / (name + '.tmp')
        temp.write_text(text)
        temp.replace(args.out / name)
    print(json.dumps({'training_rows': len(training), 'validation_rows': len(validation),
                      'files': len(outputs), 'bytes': sum(len(x.encode()) for x in outputs.values())}, indent=2))


if __name__ == '__main__':
    main()
