"""Public-only probe targets, independent per-game RNG, complete-game objectives."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from peft import get_peft_model_state_dict
from .environment import Game, Decision, seed_for
from .rollout import Step
from .endgame_feedback import EndgameFeedback


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(policy, reference=False):
    adapter = 'reference' if reference else 'policy'
    state = get_peft_model_state_dict(policy.lm, adapter_name=adapter)
    head = policy.reference_head if reference else policy.head
    state = {**state, **{'head.' + k: v for k, v in head.state_dict().items()}}
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class ProbeFeedback(EndgameFeedback):
    """Only accepts Decision (history and words), never the actual secret."""
    def describe(self, decision):
        row = super().describe(decision)
        row['probe_eligible'] = False
        row['optimal_mask'] = np.zeros(64, dtype=bool)
        if row['turn'] == 5:
            assert row['consistent'].sum() == 1, 'Frozen candidate policy requires one consistent option'
            q = row['values']
            row['optimal_mask'] = q >= q.max() - 1e-12
            row['probe_eligible'] = bool(row['n'] > 2 and
                q.max() > q[row['consistent']].max() + 1e-12)
        return row


def rloo_advantages(rewards, group):
    r = np.asarray(rewards, dtype=np.float64)
    if group < 2 or r.size % group:
        raise ValueError('Complete same-secret groups required')
    r = r.reshape(-1, group)
    advantage=r-(r.sum(1,keepdims=True)-r)/(group-1)
    # Decimal shaped rewards can leave roundoff residuals for homogeneous groups.
    advantage[np.ptp(r,axis=1)<=1e-12]=0.
    return advantage.reshape(-1)


def loss_components(log_p, log_p_sample, action, old_logp, advantage, reference, optimal_mask, clip=.2):
    """Raw T1 distributions for KL/probe; matched tempered distributions for PPO."""
    ratio = (log_p_sample[0, action] - old_logp).exp()
    pg = -torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage)
    lp, lq = log_p.double(), reference.to(log_p.device).double()
    # Generalized categorical KL removes numerical residual derivatives at equality.
    kl = (lp.exp() * (lp - lq - 1) + lq.exp()).sum().to(log_p.dtype)
    mask = torch.as_tensor(optimal_mask, dtype=torch.bool, device=log_p.device)
    probe = -torch.logsumexp(log_p[0, mask], 0) if bool(mask.any()) else log_p.sum() * 0
    return {'pg': pg, 'kl': kl, 'probe': probe, 'ratio': ratio,
            'entropy': -(log_p.exp() * log_p).sum(),
            'clipped': ((ratio.detach()-1).abs() > clip).float()}


@dataclass
class RecordedStep(Step):
    raw_log_probs: torch.Tensor | None = None
    sample_log_probs: torch.Tensor | None = None


def play_complete(policy, engine, sampler, specs, batch_size=1, temperature=1.2,
                  with_reference=False, greedy=False, progress=None):
    """specs have separate deterministic environment/action seeds for every game."""
    games = [Game(engine, sampler, s['secret'], s['environment_seed']) for s in specs]
    rngs = [np.random.default_rng(s['action_seed']) for s in specs]
    steps, active = [], list(range(len(games)))
    while active:
        decisions = [games[i].decision() for i in active]
        # One policy backbone forward returns raw logits, reusable at both temperatures.
        raw = policy.score(decisions, batch_size, 1.0)
        sample = torch.log_softmax(raw / temperature, -1)
        ref = None
        if with_reference:
            with policy.reference():
                ref = policy.score(decisions, batch_size, 1.0)
        for row, game_id in enumerate(active):
            p = sample[row].double().exp().numpy()
            action = int(raw[row].argmax()) if greedy else int(rngs[game_id].choice(64, p=p/p.sum()))
            steps.append(RecordedStep(game_id, decisions[row], action, float(sample[row, action]),
                None if ref is None else ref[row].clone(), raw[row].clone(), sample[row].clone()))
            games[game_id].step(action)
        if progress:
            progress(turn=max(len(g.history) for g in games), decisions=len(active),
                     finished=sum(g.done for g in games))
        active = [i for i in active if not games[i].done]
    assert all(g.done and (g.won or len(g.history) == 6) for g in games)
    return games, steps


def game_specs(secrets, seed, iteration, group):
    return [{'secret': str(secret), 'group_index': i, 'repeat': j,
             'environment_seed': seed_for('rl-environment', seed, iteration, str(secret), j),
             'action_seed': seed_for('rl-actions', seed, iteration, str(secret), j)}
            for i, secret in enumerate(secrets) for j in range(group)]


def records_from_games(games, steps, specs, feedback=None):
    assert all(g.done for g in games), 'No probe targets before game completion'
    records = [{**spec, 'game': i, 'won': bool(g.won), 'reward': g.terminal_reward(),
                'history': [list(x) for x in g.history], 'steps': []}
               for i, (g, spec) in enumerate(zip(games, specs))]
    for step in steps:
        row = {**step.decision.to_dict(), 'action': step.action,
               'chosen_word': step.decision.words[step.action],
               'old_sample_log_probability': step.log_prob,
               'raw_log_probabilities': step.raw_log_probs.tolist(),
               'sample_log_probabilities': step.sample_log_probs.tolist()}
        if step.ref_log_probs is not None:
            row['reference_log_probabilities'] = step.ref_log_probs.tolist()
        if feedback:
            desc = feedback.describe(step.decision)
            row.update(remaining_answers=desc['n'], probe_eligible=desc['probe_eligible'])
            if desc['turn'] == 5:
                row.update(q5=desc['values'].tolist(), optimal_indices=np.flatnonzero(desc['optimal_mask']).tolist())
        records[step.game]['steps'].append(row)
    return records


def secret_schedule(train, hard_pool, hard_fraction, seed, iteration):
    """C/A uniform schedules match; D/B uniform half is a C/A subset."""
    uniform = np.random.default_rng(seed_for('uniform-secrets', seed, iteration)).choice(train, 16, replace=False).tolist()
    if hard_fraction == 0:
        return uniform, ['uniform'] * 16
    assert hard_fraction == .5
    hard_available = sorted(set(hard_pool) - set(uniform[:8]))
    assert len(hard_available) >= 8
    hard = np.random.default_rng(seed_for('hard-secrets', seed, iteration)).choice(hard_available, 8, replace=False).tolist()
    return uniform[:8] + hard, ['uniform'] * 8 + ['hard'] * 8
