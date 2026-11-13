import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from wordle_k64.baselines import NativeLabelPolicy
from wordle_k64.checkpoint import save_checkpoint, seed_everything
from wordle_k64.contracts import load_contract
from wordle_k64.environment import Game
from wordle_k64.model import build_policy
from wordle_k64.probe_study import digest

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args):
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    result = subprocess.run([sys.executable, str(ROOT/'scripts'/script), *map(str, args)],
                            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


def test_frozen_native_readout_has_no_adapter_and_one_forward(tiny, env):
    engine, sampler = env
    policy = NativeLabelPolicy(tiny, 'cpu', 'float32')
    decision = Game(engine, sampler, load_contract()[0]['validation'][0], 12).decision()
    versions = policy.frozen_parameter_versions()
    calls, hidden = [], []
    def capture(module, args, output):
        calls.append(1)
        hidden.append(output.last_hidden_state[:, -1])
    hook = policy.lm.model.register_forward_hook(capture)
    actual = policy.score([decision])
    hook.remove()
    with torch.inference_mode():
        expected = policy.lm.get_output_embeddings()(hidden[0])[:, policy.label_ids].log_softmax(-1)
    assert len(calls) == 1 and torch.equal(actual, expected)
    assert policy.frozen_parameter_versions() == versions
    assert all(not p.requires_grad for p in policy.lm.parameters())


def test_data_generator_closes_public_singletons(tmp_path, env):
    from wordle_k64.teacher import load_episodes
    from wordle_k64.closing import closing_target
    from wordle_k64.environment import Decision
    engine, sampler = env
    out = tmp_path/'episodes'
    run('make_data.py', '--out', out, '--limit', 2, '--variants', 2, '--workers', 1)
    episodes = load_episodes(out/'train.jsonl', 'train', engine, sampler)
    assert len(episodes) == 4
    closed = 0
    for episode in episodes:
        first = episode['steps'][0]
        assert first['words'][first['action']] == 'salet'
        for step in episode['steps']:
            target = closing_target(Decision.from_dict(step), engine)
            if target is not None:
                assert step['accepted'] == [target]
                closed += 1
    assert closed > 0


def test_checkpoint_allows_local_backbone_override(tiny, tmp_path):
    policy = build_policy(tiny, 'cpu', 'float32', rank=4, alpha=8, architecture='pointer')
    checkpoint = tmp_path/'checkpoint'
    save_checkpoint(policy, checkpoint)
    metadata = json.loads((checkpoint/'metadata.json').read_text())
    metadata['model_spec']['model_id'] = 'unavailable-on-this-machine'
    (checkpoint/'metadata.json').write_text(json.dumps(metadata))
    restored = build_policy(tiny, 'cpu', 'float32', checkpoint=checkpoint)
    assert digest(policy) == digest(restored)


@pytest.mark.parametrize('experiment', ['PSL-head', 'PSL-head-consloss', 'LM-head'])
def test_sft_recipe_checkpoint_and_resume(tiny, tmp_path, experiment):
    out = tmp_path/experiment
    args = ['--experiment', experiment, '--model', tiny, '--placement', 'cpu',
            '--dtype', 'float32', '--rank', 4, '--alpha', 8,
            '--data', ROOT/'tests/train.jsonl', '--diagnostic-data', ROOT/'tests/validation.jsonl',
            '--eval-secrets', 2, '--eval-seeds', 1001, '--eval-batch', 2,
            '--panel-limit', 2, '--diagnostic-limit', 2, '--eval-every', 1, '--out', out]
    run('train_sft.py', *args, '--max-steps', 1)
    run('train_sft.py', *args, '--max-steps', 2, '--resume', out/'last')
    for step in [0, 1, 2]:
        assert (out/f'checkpoints/step_{step:05d}/trainer_state.pt').exists()
    log = [json.loads(line) for line in (out/'log.jsonl').read_text().splitlines()]
    assert [r['step'] for r in log if r['stage'] == 'sl'] == [1, 2]
    assert all('time' not in row and 'timestamp' not in row for row in log)


def test_current_evaluator_output_supports_calibration(tiny, tmp_path):
    policy = build_policy(tiny, 'cpu', 'float32', rank=4, alpha=8, architecture='pointer')
    checkpoint = tmp_path/'checkpoint'; save_checkpoint(policy, checkpoint)
    out = tmp_path/'evaluation'
    run('evaluate.py', '--policy', 'checkpoint', '--checkpoint', checkpoint,
        '--placement', 'cpu', '--dtype', 'float32', '--limit', 2,
        '--seeds', 1001, '--batch-size', 2, '--out', out)
    run('calibration.py', '--evaluation', out, '--output-prefix', tmp_path/'calibration')
    report = json.loads((tmp_path/'calibration.json').read_text())
    expected = json.loads((out/'metrics.json').read_text())
    assert report['metrics']['closing_aware_best']['all']['ece'] == pytest.approx(expected['calibration']['ECE'])
    assert report['games'] == 2


@pytest.mark.parametrize('arm', ['RL-A', 'RL-B'])
def test_rl_complete_game_updates_validation_checkpoint_and_resume(tiny, tmp_path, arm):
    seed_everything(123)
    policy = build_policy(tiny, 'cpu', 'float32', rank=4, alpha=8, architecture='pointer')
    initial = tmp_path/'initial'
    save_checkpoint(policy, initial)
    pool = json.loads((ROOT/'data/hard_train_secrets.json').read_text())
    pool['policy_sha256'] = digest(policy)
    pool_file = tmp_path/'pool.json'
    pool_file.write_text(json.dumps(pool))
    out = tmp_path/'run'
    args = ['--arm', arm, '--init', initial, '--hard-pool', pool_file, '--out', out,
            '--placement', 'cpu', '--dtype', 'float32', '--smoke']
    run('train_rl.py', *args, '--stop-after', 1)
    run('train_rl.py', *args, '--stop-after', 2, '--resume', out/'last')
    for iteration in [0, 1, 2]:
        assert (out/f'checkpoints/iteration_{iteration:05d}/trainer_state.pt').exists()
        report = json.loads((out/f'validation/iteration_{iteration:05d}/metrics.json').read_text())
        assert report['games'] == 2
        assert report['order']['comparisons'] == 3*report['decisions']
    metadata = json.loads((out/'last/metadata.json').read_text())
    assert all(name.startswith('wordle_k64/') or name in {'scripts/train_rl.py','scripts/_common.py'}
               for name in metadata['training_config']['source_sha256'])
    assert metadata['progress']['completed_games'] == 8
    assert metadata['progress']['optimizer_step'] == 2
    rollout = json.loads((out/'rollouts/iteration_00002.json').read_text())
    assert len(rollout['records']) == 4
    assert all(r['won'] or len(r['history']) == 6 for r in rollout['records'])
    assert (bool(rollout['low_information'])) == (arm == 'RL-B')
