from dataclasses import dataclass
import math
import numpy as np
import torch
from .environment import Decision, Game, seed_for
from .closing import closing_target


@dataclass
class Step:
    game: int
    decision: Decision
    action: int
    log_prob: float
    ref_log_probs: torch.Tensor | None = None


def play_games(policy, engine, sampler, secrets, rng, batch_size=8,
               greedy=False, with_reference=False, temperature=1.0):
    games = [Game(engine, sampler, s, int(rng.integers(2**63))) for s in secrets]
    active = list(range(len(games)))
    steps = []
    while active:
        decisions = [games[i].decision() for i in active]
        log_p = policy.score(decisions, batch_size, temperature)
        ref = None
        if with_reference:
            with policy.reference():
                ref = policy.score(decisions, batch_size, temperature)
        for row, game_id in enumerate(active):
            p = log_p[row].double().exp().numpy()
            action = int(p.argmax()) if greedy else int(rng.choice(64, p=p / p.sum()))
            steps.append(Step(game_id, decisions[row], action, float(log_p[row, action]),
                              None if ref is None else ref[row].clone()))
            games[game_id].step(action)
        active = [i for i in active if not games[i].done]
    assert all(g.done for g in games)
    return games, steps


def group_advantages(rewards, group):
    if group < 2 or len(rewards) % group:
        raise ValueError("Groups need >=2 complete episodes and exact group boundaries")
    r = np.asarray(rewards, dtype=np.float64).reshape(-1, group)
    mean, std = r.mean(1, keepdims=True), r.std(1, keepdims=True)
    return np.where(std > 1e-8, (r - mean) / (std + 1e-6), 0).reshape(-1)


def summarize(games):
    wins = [len(g.history) for g in games if g.won]
    return {"games": len(games), "wins": len(wins), "win_rate": len(wins) / len(games),
            "mean_guesses_wins": float(np.mean(wins)) if wins else None}


def evaluate_policy(policy, engine, sampler, secrets, seeds=(1001,), batch_size=8, greedy=True,
                    closing_metrics=False):
    reports, paired = [], np.zeros((len(seeds), len(secrets)), dtype=np.float64)
    closing_counts = {str(t): {'opportunities': 0, 'correct': 0} for t in range(1, 7)}
    for j, seed in enumerate(seeds):
        games, steps = play_games(policy, engine, sampler, secrets, np.random.default_rng(seed),
                              batch_size=batch_size, greedy=greedy)
        if closing_metrics:
            for step in steps:
                target = closing_target(step.decision, engine)
                if target is not None:
                    row = closing_counts[str(len(step.decision.history)+1)]
                    row['opportunities'] += 1
                    row['correct'] += int(target == step.action)
        paired[j] = [g.won for g in games]
        reports.append({"seed": seed, **summarize(games)})
    # Bootstrap answers as clusters, preserving repeats of the same secret together.
    per_answer = paired.mean(0)
    draws = np.random.default_rng(722).choice(per_answer, size=(2000, len(per_answer)), replace=True).mean(1)
    result = {"win_rate": float(paired.mean()), "unique_answers": len(secrets),
            "games": int(paired.size), "by_seed": reports,
            "answer_bootstrap_95pct": np.quantile(draws, [.025, .975]).tolist(),
            "per_answer_success": per_answer.tolist()}
    if closing_metrics:
        total = sum(r['opportunities'] for r in closing_counts.values())
        correct = sum(r['correct'] for r in closing_counts.values())
        for row in closing_counts.values():
            row['accuracy'] = row['correct']/row['opportunities'] if row['opportunities'] else None
        result['closing'] = {'opportunities': total, 'correct': correct,
                             'accuracy': correct/total if total else None, 'by_turn': closing_counts}
    return result
