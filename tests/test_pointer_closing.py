import json
from pathlib import Path
import numpy as np
import pytest
import torch
from wordle_k64.checkpoint import seed_everything, save_checkpoint
from wordle_k64.closing import closing_target
from wordle_k64.environment import Decision, Game
from wordle_k64.model import build_policy, POINTER_MARKERS


def test_pointer_single_forward_padding_gradients_and_checkpoint(tiny, env, tmp_path):
    engine, sampler = env
    seed_everything(73)
    policy = build_policy(tiny, placement='cpu', dtype='float32', rank=4, alpha=8,
                          head_width=32, head_seed=91, architecture='pointer')
    game = Game(engine, sampler, 'crane', 19)
    first = game.decision()
    game.step(first.words.index('salet'))
    second = game.decision()
    prompt = policy.prompts([second])[0]
    assert all((POINTER_MARKERS[2]+w.upper()+POINTER_MARKERS[3]) in prompt for w in second.words)
    assert prompt.endswith(POINTER_MARKERS[-1])
    calls = []
    handle = policy.backbone.register_forward_hook(lambda *a: calls.append(1))
    batched = policy.score([first, second])
    assert len(calls) == 1
    handle.remove()
    singles = torch.cat([policy.score([d]) for d in [first, second]])
    assert torch.allclose(batched, singles, atol=2e-5)
    assert batched.shape == (2, 64) and torch.allclose(batched.exp().sum(-1), torch.ones(2))
    policy.anchor_reference()
    with policy.reference(): reference = policy.score([second])
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=1e-3)
    policy.train()
    loss = -policy.log_probs([first, second])[:, 3].mean()
    loss.backward()
    assert policy.head.key.weight.grad.abs().sum() > 0
    assert policy.head.query.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in policy.named_parameters() if '.policy.' in n)
    torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), 1.0)
    optimizer.step()
    with policy.reference(): assert torch.equal(reference, policy.score([second]))
    path = tmp_path/'pointer'
    save_checkpoint(policy, path)
    loaded = build_policy(checkpoint=path, placement='cpu', dtype='float32')
    assert loaded.model_spec['architecture'] == 'contextual-pointer-v1'
    assert torch.allclose(loaded.score([first, second]), policy.score([first, second]), atol=1e-6)
    with loaded.reference(): assert torch.equal(reference, loaded.score([second]))
    with pytest.raises(ValueError, match='architecture'):
        build_policy(checkpoint=path, architecture='char', placement='cpu')


def test_closing_label_is_public_and_does_not_assume_the_word_is_offered(env):
    engine, sampler = env
    history = (('salet','BBBYB'),('jerky','BYYBB'),('where','BBBYG'),('grope','GGGBG'))
    state = engine.state(history)
    d = Decision(history, tuple(engine.guesses[g] for g in sampler.sample(state)))
    target = closing_target(d, engine)
    assert target is not None and d.words[target] == 'grove'
    moved, perm = d.shuffled(np.random.default_rng(11))
    assert moved.words[closing_target(moved, engine)] == d.words[target]
    words = list(d.words)
    words[target] = next(w for w in engine.guesses if w not in d.words)
    assert closing_target(Decision(d.history, tuple(words)), engine) is None
    root = Game(engine, sampler, 'crane', 22).decision()
    assert closing_target(root, engine) is None
