"""Full-game validation and public turn-five diagnostics, isolated from training."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .rollout import play_games, summarize
from .endgame_feedback import trajectory_metrics


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def evaluate_games(policy, engine, sampler, secrets, seeds, batch_size, feedback):
    games, steps, reports, successes, records = [], [], [], [], []
    for seed in seeds:
        current_games, current_steps = play_games(
            policy, engine, sampler, secrets, np.random.default_rng(seed),
            batch_size=batch_size, greedy=True)
        assert all(game.done for game in current_games)
        offset = len(games)
        games.extend(current_games)
        steps.extend(replace(step, game=step.game + offset) for step in current_steps)
        reports.append({'seed': seed, **summarize(current_games)})
        successes.append([game.won for game in current_games])
        grouped = [[] for _ in current_games]
        for step in current_steps:
            grouped[step.game].append({**step.decision.to_dict(), 'action': step.action,
                'chosen_word': step.decision.words[step.action], 'log_probability': step.log_prob})
        for index, game in enumerate(current_games):
            records.append({'seed': seed, 'answer_index': index, 'answer': secrets[index],
                'won': game.won, 'history': game.history, 'decisions': grouped[index]})
    per_answer = np.asarray(successes, dtype=float).mean(0)
    draws = np.random.default_rng(722).choice(
        per_answer, size=(2000, len(per_answer)), replace=True).mean(1)
    result = {**summarize(games), 'unique_answers': len(secrets), 'by_seed': reports,
        'answer_bootstrap_95pct': np.quantile(draws, [.025, .975]).tolist(),
        'per_answer_success': per_answer.tolist(),
        'guess_histogram': {str(t): sum(g.won and len(g.history) == t for g in games)
                            for t in range(1, 7)},
        'failed_games': sum(not g.won for g in games),
        **trajectory_metrics(games, steps, feedback)}
    turn5 = result['turn5']
    available = turn5.get('guarantee_available_count', 0)
    turn5['missed_guarantee_rate'] = turn5.get('guarantee_missed_count', 0) / available if available else None
    return result, records
