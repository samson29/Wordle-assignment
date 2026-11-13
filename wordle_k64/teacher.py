"""Complete teacher episodes, without exposing offline secrets to the policy."""
from pathlib import Path
import json
import numpy as np
from .contracts import fingerprint, load_contract, require_fingerprint
from .environment import Decision, Game, feedback, seed_for
from .planner import ChallengePlanner


def generate_episode(engine, sampler, planner, answer, split, index, seed):
    game = Game(engine, sampler, answer, seed_for("teacher-order", seed, split, index))
    steps = []
    while not game.done:
        decision = game.decision()
        selected, rows = planner.choose(engine.state(game.history), game.options)
        q = next(r["continuation_win_probability"] for r in rows if r["guess_id"] == selected)
        accepted_ids = {r["guess_id"] for r in rows if abs(r["continuation_win_probability"] - q) < 1e-12}
        accepted = [i for i, g in enumerate(game.options) if g in accepted_ids]
        action = game.options.index(selected)
        result = game.step(action)
        steps.append({**decision.to_dict(), "action": action, "accepted": accepted,
                      "feedback": result["feedback"], "teacher_value": q,
                      "label_kind": "certified_win" if q >= 1 - 1e-12 else "fixed_continuation",
                      "all_actions_evaluated": len(rows) == 64})
    return {"episode_id": f"{split}-{index:05d}", "split": split, "contract": fingerprint(),
            "offline_answer": answer, "won": game.won, "reward": game.terminal_reward(), "steps": steps}


def load_episodes(path, split, engine=None, sampler=None):
    if split not in {"train", "validation"}:
        raise ValueError("Teacher data may only use train/validation episodes")
    manifest, _ = load_contract()
    allowed = set(manifest[split])
    expected_contract = fingerprint()
    episodes, seen = [], set()
    for line in Path(path).read_text().splitlines():
        item = json.loads(line)
        if item["contract"] != expected_contract:
            raise ValueError("Teacher data belongs to a different experiment contract")
        if item["split"] != split or item["offline_answer"] not in allowed:
            raise ValueError("Teacher episode crosses the frozen split")
        if item["episode_id"] in seen:
            raise ValueError("Duplicate episode ID")
        seen.add(item["episode_id"])
        history = ()
        for step in item["steps"]:
            d = Decision.from_dict(step)
            if d.history != history:
                raise ValueError("Incomplete or noncausal teacher trajectory")
            a, accepted = int(step["action"]), step["accepted"]
            if not 0 <= a < 64 or not accepted or any(not 0 <= i < 64 for i in accepted) or a not in accepted:
                raise ValueError("Invalid teacher actions")
            pattern = feedback(d.words[a], item["offline_answer"])
            if pattern != step["feedback"]:
                raise ValueError("Teacher feedback mismatch")
            if engine is not None:
                if any(w not in engine.gid for w in d.words):
                    raise ValueError("Illegal candidate word")
                if sampler is not None:
                    expected = {engine.guesses[g] for g in sampler.sample(engine.state(history))}
                    if expected != set(d.words):
                        raise ValueError("Teacher candidate policy mismatch")
            history += ((d.words[a], pattern),)
        won = bool(history) and history[-1][1] == "GGGGG"
        if not history or len(history) > 6 or (not won and len(history) != 6):
            raise ValueError("A teacher record must be a complete game")
        if item["won"] != won or item["reward"] != float(won):
            raise ValueError("Terminal outcome mismatch")
        episodes.append(item)
    if not episodes:
        raise ValueError("Empty episode file")
    return episodes


def flatten_episode_batch(episodes, rng):
    decisions, masks = [], []
    for episode in episodes:
        for step in episode["steps"]:
            decision, perm = Decision.from_dict(step).shuffled(rng)
            decisions.append(decision)
            masks.append(np.isin(perm, step["accepted"]))
    return decisions, np.stack(masks)
