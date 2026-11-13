"""Recount released held-out metrics and confidence intervals without model inference.

Default: verify the compact bundled observations against published tables.
--replay: also replay every game against the frozen environment (CPU only).
--checkpoints: hash every released checkpoint file, including optimizer states.
--archive FINAL: maintainer-only rebuild of the observation bundle from raw files.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from _common import ROOT
from calibration import calculate
from wordle_k64.contracts import load_contract, sha256

SEEDS = [1001, 1002, 1003]
SOURCES = {
    'heuristic_random_opening': 'baselines/heuristic_random_opening',
    'heuristic_distinct_opening': 'baselines/heuristic_distinct_opening',
    'heuristic_salet': 'baselines/heuristic_salet',
    'random_policy': 'baselines/random_policy',
    'zero_shot_qwen': 'baselines/zero_shot_qwen',
    'PSL-head@1100': 'heldout/step1100/PSL-head',
    'PSL-head-consloss@1100': 'heldout/step1100/PSL-head-consloss',
    'RL-A@12': 'heldout/rl_iteration12/D',
    'RL-B@12': 'heldout/rl_iteration12/E',
}
BUNDLE = ROOT/'results/heldout_samples.jsonl.gz'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name):
    if actual is None or expected is None:
        require(actual == expected, f'{name}: {actual} != {expected}')
    else:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, err_msg=name)


def build_bundle(archive):
    """Keep trajectories and per-decision sufficient statistics; omit dense logits."""
    from _common import environment
    from wordle_k64.environment import Decision
    from wordle_k64.probe_study import ProbeFeedback
    _, engine, sampler = environment()
    feedback = ProbeFeedback(engine)
    hashes, records = {}, []

    def read(path):
        hashes[str(path.relative_to(archive))] = sha256(path)
        return json.loads(path.read_text())

    for model, relative in SOURCES.items():
        folder = archive/relative
        cpu = (folder/'games.jsonl').exists()
        if cpu:
            path = folder/'games.jsonl'
            hashes[str(path.relative_to(archive))] = sha256(path)
            games = [json.loads(line) for line in path.read_text().splitlines()]
            labels = {}
        else:
            paths = sorted(folder.glob('games_seed*.json')) or sorted(folder.glob('shards/*/games_seed*.json'))
            require(len(paths) == 3, f'{model}: three game files required')
            games = [g for p in paths for g in read(p)]
            row_paths = [folder/'decision_analysis.json'] if (folder/'decision_analysis.json').exists() else [folder/'decision_metrics.json']
            if model == 'zero_shot_qwen': row_paths = sorted(folder.glob('analysis_seed*.json'))
            rows = [r for p in row_paths for r in read(p)]
            labels = {(r['seed'], r['answer_index'], r['turn']): r for r in rows}
            require(len(labels) == len(rows), 'Duplicate decision labels')
        by_key = {(g['seed'], g['answer_index'], t): s
                  for g in games for t, s in enumerate(g['steps'], 1)}
        orders = defaultdict(dict)
        if not cpu:
            paths = sorted(folder.glob('order/*.npz')) or sorted(folder.glob('shards/*/order/*.npz'))
            require(len(paths) == 12, f'{model}: 12 order files required')
            for path in paths:
                hashes[str(path.relative_to(archive))] = sha256(path)
                seed, phase = path.stem.split('_', 1)
                with np.load(path) as data:
                    lp = data['log_probs'] if 'log_probs' in data else data['log_probabilities']
                    perms = data['permutations']
                    for i, (answer, turn) in enumerate(zip(data['answer_index'], data['turn'])):
                        key = (int(seed), int(answer), int(turn)); step = by_key[key]
                        raw = np.asarray(step.get('log_probabilities', step.get('raw_log_probabilities')), dtype=float)
                        require(int(raw.argmax()) == step['action'], 'Non-greedy archived action')
                        p = np.exp(raw)
                        q = np.exp(lp[i].astype(float))[np.argsort(perms[i])]
                        flip = int(perms[i, lp[i].argmax()] != step['action'])
                        tv = float(np.abs(p-q).sum()/2)
                        close(flip, data['flips'][i], f'{model} saved flip')
                        close(tv, data['tv'][i], f'{model} saved TV')
                        orders[key][phase] = [flip, tv]
        for game in sorted(games, key=lambda g: (g['seed'], g['answer_index'])):
            record = {'model': model, 'seed': game['seed'], 'answer_index': game['answer_index'],
                      'secret': game.get('secret', game.get('offline_secret')),
                      'environment_seed': game.get('environment_seed', game.get('game_seed')),
                      'won': bool(game['won']), 'steps': []}
            history = []
            for turn, step in enumerate(game['steps'], 1):
                require(step['history'] == history, 'Broken causal history')
                pattern = step.get('feedback') or game['history'][turn-1][1]
                action = step['action']; word = step['words'][action]
                require(word == step['chosen_word'], 'Action/word mismatch')
                item = {'word': word, 'feedback': pattern, 'action': action}
                if not cpu:
                    key = (game['seed'], game['answer_index'], turn); row = labels[key]
                    p = np.exp(np.asarray(step.get('log_probabilities', step.get('raw_log_probabilities')), dtype=float))
                    require(abs(p.sum()-1) < 1e-5, 'Bad probability distribution')
                    close(p[action], row['confidence'], 'Confidence')
                    item.update({k: row[k] for k in ['confidence', 'chosen_q', 'best_q', 'closing_aware_best',
                        'closing_opportunity', 'closing_correct']})
                    item['n'] = row.get('n', row.get('remaining_count'))
                    desc = feedback.describe(Decision.from_dict(step))
                    require(desc['n'] == item['n'], 'Posterior count mismatch')
                    item['probe_required'] = int(desc['probe_eligible'])
                    item['probe_optimal'] = int(desc['probe_eligible'] and row['chosen_q'] >= row['best_q']-1e-12)
                    item['guarantee_available'] = int(turn == 5 and item['n'] > 2 and row['best_q'] >= 1-1e-12)
                    item['guarantee_missed'] = int(item['guarantee_available'] and row['chosen_q'] < 1-1e-12)
                    item['order'] = orders[key]
                    require(set(item['order']) == {'repeat','reverse','random1','random2'}, 'Incomplete order comparisons')
                record['steps'].append(item)
                history.append([word, pattern])
            records.append(record)
        print(f'Bundled {model}: {len(games)} games', flush=True)
    with BUNDLE.open('wb') as stream:
        with gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as zipped:
            for record in records:
                zipped.write((json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n').encode())
    provenance = json.loads((ROOT/'results/provenance.json').read_text())
    provenance['heldout_observations'] = {'path': str(BUNDLE.relative_to(ROOT)), 'sha256': sha256(BUNDLE),
        'games': len(records), 'source_sha256': hashes,
        'contents': 'Complete guess/feedback trajectories, selected confidence, public solver labels, and each word-aligned order comparison. Dense model logits omitted; rerun evaluate.py to regenerate them.'}
    (ROOT/'results/provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


def ratio(rows, numerator, denominator, weights):
    clusters = np.zeros((500, 2))
    for row in rows:
        clusters[row['answer_index']] += [numerator(row), denominator(row)]
    totals = clusters.sum(0); draws = weights @ clusters
    valid = draws[:, 1] > 0
    return {'numerator': float(totals[0]), 'denominator': int(totals[1]),
            'mean': float(totals[0]/totals[1]) if totals[1] else None,
            'answer_bootstrap_95pct': np.quantile(draws[valid,0]/draws[valid,1], [.025,.975]).tolist() if valid.any() else None}


def verify(replay=False):
    split, _ = load_contract()
    provenance = json.loads((ROOT/'results/provenance.json').read_text())
    require(sha256(BUNDLE) == provenance['heldout_observations']['sha256'], 'Observation bundle hash mismatch')
    if replay:
        from _common import environment
        from wordle_k64.environment import Game
        from wordle_k64.probe_study import ProbeFeedback
        _, engine, sampler = environment(); feedback = ProbeFeedback(engine)
    with gzip.open(BUNDLE, 'rt') as handle:
        games = [json.loads(line) for line in handle]
    grouped = defaultdict(list)
    for game in games: grouped[game['model']].append(game)
    require(set(grouped) == set(SOURCES), 'Model coverage differs')
    draws = np.random.default_rng(722).integers(0, 500, size=(2000,500))
    weights = np.asarray([np.bincount(d, minlength=500) for d in draws], dtype=float)
    expected = json.loads((ROOT/'results/heldout_metrics.json').read_text())
    summary = {}
    for model, gs in grouped.items():
        require(len(gs) == 1500 and {(g['seed'],g['answer_index']) for g in gs} == {(s,i) for s in SEEDS for i in range(500)}, f'{model}: game coverage')
        rows = []
        for game in gs:
            require(split['test'][game['answer_index']] == game['secret'], 'Incorrect held-out answer')
            require(1 <= len(game['steps']) <= 6, 'Bad game length')
            require(game['won'] == (game['steps'][-1]['feedback'] == 'GGGGG'), 'Incorrect win flag')
            require(game['won'] or len(game['steps']) == 6, 'Incomplete loss')
            require(all(s['feedback'] != 'GGGGG' for s in game['steps'][:-1]), 'Game continued after winning')
            if replay: env_game = Game(engine, sampler, game['secret'], game['environment_seed'])
            for turn, step in enumerate(game['steps'],1):
                if replay:
                    decision = env_game.decision()
                    require(decision.words[step['action']] == step['word'], 'Candidate order/action mismatch')
                    if 'confidence' in step:
                        desc = feedback.describe(decision)
                        require(desc['n'] == step['n'], 'Incorrect public posterior')
                        require(desc['probe_eligible'] == bool(step['probe_required']), 'Incorrect probe eligibility')
                        require((desc['singleton'] is not None) == bool(step['closing_opportunity']), 'Incorrect closing opportunity')
                        if step['closing_opportunity']: require((step['action'] == desc['singleton']) == bool(step['closing_correct']), 'Incorrect closing label')
                        if turn == 5:
                            close(desc['values'].max(), step['best_q'], 'Exact turn-five best value')
                            close(desc['values'][step['action']], step['chosen_q'], 'Exact turn-five selected value')
                    result = env_game.step(step['action'])
                    require(result['feedback'] == step['feedback'], 'Incorrect feedback')
                if 'confidence' in step:
                    correct = step['closing_correct'] if step['closing_opportunity'] else int(step['chosen_q'] >= step['best_q']-1e-12)
                    require(correct == step['closing_aware_best'], 'Incorrect calibration label')
                    require(0 <= step['confidence'] <= 1 and 0 <= step['chosen_q'] <= step['best_q']+1e-10 <= 1+1e-10, 'Invalid probabilities')
                    rows.append({'answer_index':game['answer_index'], 'seed':game['seed'], 'turn':turn, **step})
        full_report = expected[model]
        report = full_report.get('performance', full_report)
        wins = ratio(gs, lambda g:g['won'], lambda g:1, weights)
        guesses = ratio(gs, lambda g:len(g['steps'])*g['won'], lambda g:g['won'], weights)
        close(wins['numerator'], report['wins'], model+' wins')
        close(wins['mean'], report['win_rate'], model+' win rate')
        close(wins['answer_bootstrap_95pct'], report.get('answer_bootstrap_95pct', report.get('win_rate_95pct')), model+' win CI')
        old_mean = report['mean_guesses_wins']
        close(guesses['mean'], old_mean['mean'] if isinstance(old_mean,dict) else old_mean, model+' mean guesses')
        close(guesses['answer_bootstrap_95pct'], old_mean['answer_bootstrap_95pct'] if isinstance(old_mean,dict) else report['mean_guesses_wins_95pct'], model+' guess CI')
        summary[model] = {'games':len(gs), 'wins':int(wins['numerator']), 'win_rate':wins['mean'], 'mean_guesses_wins':guesses}
        for key, saved in report['guess_distribution'].items():
            value = ratio(gs, lambda g:(not g['won']) if key=='failed' else g['won'] and len(g['steps'])==int(key), lambda g:1, weights)
            close(value['mean'], saved.get('mean',saved.get('fraction')), model+' histogram '+key)
            close(value['answer_bootstrap_95pct'], saved.get('answer_bootstrap_95pct',saved.get('fraction_95pct')), model+' histogram CI '+key)
        if rows:
            cal = calculate(rows, 'closing_aware_best', 15, {i:i for i in range(500)}, weights)
            saved_cal = full_report.get('calibration') or full_report['endgame_and_calibration']['calibration_closing_aware']
            close(cal['ece'], saved_cal['ECE'], model+' ECE')
            close(cal['ece_95ci'], saved_cal['answer_bootstrap_95pct'], model+' ECE CI')
            order = full_report['order']
            for name, phases, index, saved in [
                ('flip_rate',['reverse','random1','random2'],0, order['flip_rate']),
                ('mean_tv',['reverse','random1','random2'],1,order.get('mean_tv',order.get('mean_TV'))),
                ('same_order_flip_rate',['repeat'],0,order.get('same_order_flip_rate',order.get('same_order',{}).get('flip_rate'))),
                ('same_order_mean_tv',['repeat'],1,order.get('same_order_mean_tv',order.get('same_order',{}).get('mean_TV'))),
            ]:
                value = ratio(rows,lambda r:sum(r['order'][p][index] for p in phases),lambda r:len(phases),weights)
                close(value['mean'], saved['mean'] if isinstance(saved,dict) else saved, model+' '+name)
                ci_key = {'flip_rate':'flip_rate_95pct','mean_tv':'mean_TV_95pct','same_order_flip_rate':'flip_rate_95pct','same_order_mean_tv':'mean_TV_95pct'}[name]
                old_ci = saved['answer_bootstrap_95pct'] if isinstance(saved,dict) else (order['same_order'] if name.startswith('same_order') else order)[ci_key]
                close(value['answer_bootstrap_95pct'], old_ci, model+' '+name+' CI')
                summary[model][name] = value
            summary[model].update(decisions=len(rows), ece=cal['ece'], ece_95ci=cal['ece_95ci'])
            for name, num, den in [('probe_required','probe_optimal','probe_required'),('missed_turn5_guarantees','guarantee_missed','guarantee_available'),('closing_with_ci','closing_correct','closing_opportunity')]:
                summary[model][name] = ratio(rows,lambda r:r[num],lambda r:r[den],weights)
        print(f'PASS {model}: {len(gs)} games, {int(wins["numerator"])} wins'+(f', {len(rows)} decisions' if rows else ''),flush=True)
    # Compare the endgame table independently, including the starting checkpoint.
    aliases = {'PSL-head-consloss @1100':'PSL-head-consloss@1100','RL-A @12':'RL-A@12','RL-B @12':'RL-B@12'}
    metric_aliases = {'optimal_probe_choice_when_required_turn5':'probe_required', 'missed_turn5_guarantees':'missed_turn5_guarantees','closing_accuracy':'closing_with_ci'}
    for row in csv.DictReader((ROOT/'results/endgame_comparison.csv').open()):
        key = metric_aliases[row['metric']]; value = summary[aliases[row['model']]][key]
        close([value['numerator'],value['denominator'],value['mean'],*value['answer_bootstrap_95pct']],
              [float(row[k]) for k in ['numerator','denominator','rate','ci_low','ci_high']], 'Endgame table')
    return summary


def checkpoint_integrity():
    manifest = json.loads((ROOT/'checkpoints/manifest.json').read_text()); count = 0
    for run, entry in manifest.items():
        for relative, expected in entry['files'].items():
            path = ROOT/entry['path']/relative
            require(path.exists(), f'Missing checkpoint file {path.relative_to(ROOT)}; run git lfs pull')
            require(path.stat().st_size == expected['bytes'] and sha256(path) == expected['sha256'], f'Checkpoint mismatch: {path.relative_to(ROOT)}')
            count += 1
    print(f'PASS {count} checkpoint files (SHA-256)',flush=True)
    return count


def verify_tables(summary):
    """Check CSV views and plot inputs, not only their parent JSON report."""
    aliases = {'PSL-head':'PSL-head@1100', 'PSL-head-consloss':'PSL-head-consloss@1100',
               'RL-A':'RL-A@12', 'RL-B':'RL-B@12', 'Qwen (zero-shot)':'zero_shot_qwen'}
    for row in csv.DictReader((ROOT/'results/order_robustness.csv').open()):
        model = summary[aliases[row['model']]]
        for metric in ['flip_rate','mean_tv','same_order_flip_rate']:
            value = model[metric]
            close([value['mean'],*value['answer_bootstrap_95pct']],
                  [float(row[k]) for k in [metric,metric+'_ci_low',metric+'_ci_high']], 'Order CSV '+metric)
    cal = json.loads((ROOT/'results/overall_calibration_comparison.json').read_text())
    for model in cal['models']:
        saved = summary[model['model'].replace(' @','@')]
        close([model['ece'],*model['ece_95ci']], [saved['ece'],*saved['ece_95ci']], 'Overall calibration figure')
    provenance = json.loads((ROOT/'results/provenance.json').read_text())
    for name, entry in provenance['exports'].items():
        require(sha256(ROOT/'results'/name) == entry['sha256'], f'Export hash differs: {name}')
    training = list(csv.DictReader((ROOT/'results/training.csv').open()))
    validation = list(csv.DictReader((ROOT/'results/validation.csv').open()))
    require(set(r['run'] for r in training) == set(provenance['runs']), 'Training run coverage')
    for run, expected in provenance['runs'].items():
        updates = [int(r['step']) for r in training if r['run']==run and r['stage'] in {'sl','update'}]
        require(updates == list(range(1, expected['last_update']+1)), f'{run}: missing or duplicate updates')
        vals = [r for r in validation if r['run']==run]
        require(len(vals) == expected['validation_count'], f'{run}: validation count differs')
        progress = 'iteration' if run.startswith('RL-') else 'step'
        expected_steps = [0,4,8,12] if progress=='iteration' else list(range(0,expected['last_validation_step']+1,50))
        require([int(r[progress]) for r in vals] == expected_steps, f'{run}: validation history differs')
    from _common import RUN_TIMESTAMP_KEYS
    for row in training+validation:
        require(not any(set(k.split('.')) & RUN_TIMESTAMP_KEYS for k in row), 'Training timestamp found')
    print(f'PASS result tables and complete histories: {len(training)} training rows, {len(validation)} validations',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,help='Maintainer: rebuild compact observations from FINAL')
    parser.add_argument('--replay',action='store_true')
    parser.add_argument('--checkpoints',action='store_true')
    args = parser.parse_args()
    if args.archive: build_bundle(args.archive)
    summary = verify(args.replay)
    verify_tables(summary)
    if args.checkpoints: checkpoint_integrity()
    print('All requested checks passed. No model inference or training was run.')


if __name__ == '__main__': main()
