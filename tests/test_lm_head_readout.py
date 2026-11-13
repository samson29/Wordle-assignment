import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from wordle_k64.checkpoint import seed_everything, save_checkpoint, resume_training
from wordle_k64.environment import Game
from wordle_k64.lm_head_readout import MARKER, verify_continuations
from wordle_k64.model import build_policy
from wordle_k64.sl_objectives import optimizer_groups
from _common import warmup


def make_policy(tiny):
    return build_policy(tiny, placement='cpu', dtype='float32', rank=4, alpha=8,
                        architecture='lm_head', head_seed=20261115)


def test_pretrained_rows_native_logits_and_single_forward(tiny, env):
    engine, sampler = env
    seed_everything(123)
    policy = make_policy(tiny)
    game = Game(engine, sampler, 'crane', 71)
    root = game.decision(); game.step(root.words.index('salet')); second = game.decision()
    for decision, text in zip([root, second], policy.prompts([root, second])):
        assert text.endswith(MARKER)
        assert 'Which word should be guessed next?' in text
        assert all(f'{label}: {word.upper()}' in text for label, word in zip(policy.labels, decision.words))
        verify_continuations(policy.tokenizer, text, policy.labels, policy.label_ids)
    native = policy.lm.get_base_model().get_output_embeddings()
    ids = torch.tensor(policy.label_ids)
    assert torch.equal(policy.head.weight, native.weight[ids].float())
    calls, hidden = [], []
    def capture(module, args, output):
        calls.append(1); hidden.append(output.last_hidden_state[:, -1].detach())
    handle = policy.backbone.register_forward_hook(capture)
    policy.eval()
    with torch.no_grad():
        logits = policy([root, second])
    handle.remove()
    assert len(calls) == 1
    # Compare with the FULL pretrained vocabulary projection, then slice labels.
    expected = native(hidden[0])[:, ids].float()
    assert torch.allclose(logits, expected, atol=1e-6)
    with policy.lm.disable_adapter():
        assert torch.equal(policy.score([root, second]), logits.log_softmax(-1))
    assert torch.isfinite(logits).all() and logits.shape == (2, 64)
    assert torch.allclose(policy.score([root, second]).exp().sum(-1), torch.ones(2))


def test_lm_rows_lora_gradients_frozen_base_reference_and_resume(tiny, env, tmp_path):
    engine, sampler = env
    rng = seed_everything(123)
    policy = make_policy(tiny)
    decision = Game(engine, sampler, 'crane', 71).decision()
    frozen = policy.lm.get_base_model().get_output_embeddings().weight.detach().clone()
    policy.anchor_reference()
    with policy.reference():
        reference = policy.score([decision])
    optimizer = torch.optim.AdamW(optimizer_groups(policy, 1e-4, 1e-4), weight_decay=0)
    scheduler = warmup(optimizer, 2)
    def update(p, opt, sched):
        p.train(); opt.zero_grad()
        loss = -p.log_probs([decision])[0, 5]
        loss.backward()
        assert p.head.weight.grad.abs().sum() > 0
        assert any(t.grad is not None and t.grad.abs().sum() > 0 for n, t in p.lm.named_parameters() if '.policy.' in n)
        assert all(t.grad is None for n, t in p.lm.named_parameters() if '.policy.' not in n)
        torch.nn.utils.clip_grad_norm_(p.trainable_parameters(), 1.)
        opt.step(); sched.step()
        return float(loss.detach())
    update(policy, optimizer, scheduler)
    assert torch.equal(policy.lm.get_base_model().get_output_embeddings().weight, frozen)
    with policy.reference():
        assert torch.equal(policy.score([decision]), reference)
    config = {'test': 'lm_head_resume'}
    save_checkpoint(policy, tmp_path/'checkpoint', optimizer=optimizer, scheduler=scheduler,
                    rng=rng, progress={'step': 1}, training_config=config)
    expected_loss = update(policy, optimizer, scheduler)
    expected_probs = policy.score([decision])
    restored = build_policy(checkpoint=tmp_path/'checkpoint', placement='cpu', dtype='float32')
    opt2 = torch.optim.AdamW(optimizer_groups(restored, 1e-4, 1e-4), weight_decay=0)
    sched2 = warmup(opt2, 2)
    assert resume_training(tmp_path/'checkpoint', opt2, sched2, rng, config)['step'] == 1
    actual_loss = update(restored, opt2, sched2)
    assert abs(actual_loss-expected_loss) < 1e-7
    assert torch.equal(restored.score([decision]), expected_probs)
    with restored.reference():
        assert torch.equal(restored.score([decision]), reference)


def test_lm_head_trainer_retains_validation_checkpoints_and_resumes(tiny, tmp_path):
    code = Path(__file__).resolve().parents[1]
    out = tmp_path/'sl'
    common = [sys.executable, str(code/'scripts/train_sft.py'), '--model', tiny,
        '--experiment', 'LM-head', '--architecture', 'lm_head', '--placement', 'cpu', '--dtype', 'float32',
        '--rank', '4', '--alpha', '8', '--seed', '123',
        '--data', str(code/'tests/train.jsonl'), '--out', str(out),
        '--games-per-step', '1', '--eval-secrets', '1', '--eval-batch', '1',
        '--eval-every', '1', '--save-every', '1',
        '--turn5-panel', str(code/'data/sft_turn5_panel.json'), '--panel-limit', '2',
        '--diagnostic-data', str(code/'tests/validation.jsonl'), '--diagnostic-limit', '7']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    for args in [['--max-steps', '1'], ['--max-steps', '2', '--resume', str(out/'last')]]:
        result = subprocess.run(common+args, cwd=code, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout+result.stderr
    for step in [0, 1, 2]:
        cp = out/'checkpoints'/f'step_{step:05d}'
        assert (cp/'trainer_state.pt').exists()
        metadata = json.loads((cp/'metadata.json').read_text())
        assert metadata['model_spec']['architecture'] == 'vocabulary-label-lm-head-v1'
        assert metadata['progress']['step'] == step
        val = json.loads((out/'validation'/f'step_{step:05d}.json').read_text())
        assert val['turn5_order']['comparisons'] == 6 and val['diagnostics']['comparisons'] == 21
        assert (out/'order_predictions'/f'step_{step:05d}.json').exists()
