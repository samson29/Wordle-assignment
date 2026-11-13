"""Complete SALET-first teacher trajectories, train/validation secrets only."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import time
from _common import ROOT, environment
from wordle_k64.contracts import fingerprint, sha256
from wordle_k64.environment import Game, canonical, seed_for
from wordle_k64.planner import ChallengePlanner
from wordle_k64.teacher import load_episodes
from wordle_k64.closing import closing_target


class SaletTeacher(ChallengePlanner):
    def __init__(self, engine, sampler):
        super().__init__(engine, sampler)
        self.rng = random.Random(0)
        self.varied = False
        self.root_value = None

    def choose(self, state, candidates):
        if not state.history:
            chosen = self.engine.gid["salet"]
            assert chosen in candidates
            if self.root_value is None:
                self.root_value = self._action_value(state, chosen, random.Random(0))
            return chosen, [{"guess_id": chosen, "continuation_win_probability": self.root_value}]
        chosen, rows = super().choose(state, candidates)
        if not self.varied or len(state.answers) == 1:
            return chosen, rows
        possible = self._possible(state)
        ranked = sorted(candidates, key=lambda g: (-round(self.engine.entropy(state, g), 12), self.engine.guesses[g]))
        pool = set(ranked[:8]) | (set(candidates) & possible) | {chosen}
        evaluated = {r["guess_id"]: r for r in rows}
        for guess in sorted(pool):
            if guess not in evaluated:
                evaluated[guess] = {"guess_id": guess,
                    "continuation_win_probability": self._action_value(state, guess, random.Random(0))}
        best = max(r["continuation_win_probability"] for r in evaluated.values())
        choices = sorted(g for g in pool if abs(evaluated[g]["continuation_win_probability"] - best) < 1e-12)
        if not choices:
            choices = sorted(g for g, r in evaluated.items() if abs(r["continuation_win_probability"] - best) < 1e-12)
        return self.rng.choice(choices), list(evaluated.values())


def initialize():
    global SPLIT, ENGINE, SAMPLER, TEACHER, CONTRACT
    SPLIT, ENGINE, SAMPLER = environment()
    TEACHER = SaletTeacher(ENGINE, SAMPLER)
    CONTRACT = fingerprint()


def trajectory(ep):
    return tuple((s["words"][s["action"]], s["feedback"]) for s in ep["steps"])


def make_episode(answer, split, index, variant, seed):
    episode_seed = seed_for("salet-teacher", seed, split, index, variant)
    game = Game(ENGINE, SAMPLER, answer, episode_seed)
    TEACHER.rng = random.Random(episode_seed)
    TEACHER.varied = variant > 0
    steps = []
    while not game.done:
        d = game.decision()
        chosen, rows = TEACHER.choose(ENGINE.state(game.history), game.options)
        q = next(r["continuation_win_probability"] for r in rows if r["guess_id"] == chosen)
        accepted_ids = {r["guess_id"] for r in rows if abs(r["continuation_win_probability"] - q) < 1e-12}
        accepted = [i for i, g in enumerate(game.options) if g in accepted_ids]
        action = game.options.index(chosen)
        result = game.step(action)
        target = closing_target(d, ENGINE)
        original_accepted = accepted
        if target is not None:
            assert target == action
            accepted = [target]
        steps.append({**d.to_dict(), "action": action, "accepted": accepted,
                      "closing_target": target,
                      **({"original_accepted": original_accepted} if original_accepted != accepted else {}),
                      "feedback": result["feedback"], "teacher_value": q,
                      "label_kind": "certified_win" if q >= 1 - 1e-12 else "fixed_continuation",
                      "all_actions_evaluated": len(rows) == 64})
    return {"episode_id": f"{split}-{index:05d}-v{variant:02d}", "split": split,
            "contract": CONTRACT, "offline_answer": answer, "won": game.won,
            "reward": game.terminal_reward(), "steps": steps,
            "teacher_policy": "salet-opening-equal-value-diverse-continuation-v1", "variant": variant,
            "supervision_rule": "public-singleton-and-offered-v1"}


def work(item):
    split, index, answer, variants, seed = item
    episodes = []
    seen = set()
    for variant in range(variants):
        # Try additional seeds for repeated histories, without manufacturing bad
        # moves after the answer is already known merely to make data unique.
        ep = None
        for attempt in range(3 if variant else 1):
            ep = make_episode(answer, split, index, variant, seed + attempt * 100003)
            if trajectory(ep) not in seen:
                break
        seen.add(trajectory(ep))
        episodes.append(ep)
    return episodes


def write_json(path, value):
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--variants", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    assert args.variants > 0 and args.workers > 0
    if args.limit is not None and args.limit <= 0:
        p.error('--limit must be positive')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "READY.json").exists():
        raise FileExistsError("Dataset already verified")
    started = time.monotonic()
    status = {"status": "generating", "partial": args.limit is not None}
    write_json(out / "generation_status.json", status)
    try:
        split, _, _ = environment()
        report = {"contract": fingerprint(), "seed": args.seed, "variants_per_training_secret": args.variants,
                  "partial": args.limit is not None, "test_episodes": 0, "teacher_opening": "salet"}
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
            for name in ("train", "validation"):
                secrets = split[name][:args.limit] if args.limit else split[name]
                variants = args.variants if name == "train" else 1
                temp = out / (name + ".jsonl.writing")
                unique_trajectories, unique_histories = set(), set()
                games = decisions = wins = 0
                turns = Counter()
                items = ((name, i, answer, variants, args.seed) for i, answer in enumerate(secrets))
                with temp.open("w") as f:
                    for i, episodes in enumerate(pool.map(work, items, chunksize=1)):
                        for ep in episodes:
                            assert ep["steps"][0]["words"][ep["steps"][0]["action"]] == "salet"
                            f.write(json.dumps(ep) + "\n")
                            games += 1; decisions += len(ep["steps"]); wins += ep["won"]
                            unique_trajectories.add((ep["offline_answer"], trajectory(ep)))
                            unique_histories.update(canonical(s["history"]) for s in ep["steps"])
                            turns[len(ep["steps"]) if ep["won"] else "failed"] += 1
                        if (i + 1) % 25 == 0 or i + 1 == len(secrets):
                            f.flush()
                            status.update(split=name, completed_secrets=i + 1, total_secrets=len(secrets),
                                          games=games, decisions=decisions, elapsed_s=time.monotonic() - started)
                            write_json(out / "generation_status.json", status)
                            print(json.dumps(status), flush=True)
                temp.replace(out / (name + ".jsonl"))
                report[name] = {"secrets": len(secrets), "games": games, "decisions": decisions,
                                "wins": wins, "unique_trajectories": len(unique_trajectories),
                                "unique_histories": len(unique_histories), "guess_histogram": dict(turns),
                                "sha256": sha256(out / (name + ".jsonl"))}
        status["status"] = "verifying"; write_json(out / "generation_status.json", status)
        # Verify complete causal histories, all feedback, split membership, and
        # exact candidate construction for both files before releasing training.
        _, engine, sampler = environment()
        expected = json.loads((ROOT/'data/teacher_manifest.json').read_text())
        for name in ("train", "validation"):
            checked = load_episodes(out / (name + ".jsonl"), name, engine, sampler)
            counts = Counter(ep["offline_answer"] for ep in checked)
            assert set(counts) == set(split[name][:args.limit] if args.limit else split[name])
            assert set(counts.values()) == {args.variants if name == "train" else 1}
            semantic = hashlib.sha256()
            for episode in checked:
                semantic.update((json.dumps(episode, sort_keys=True, separators=(',', ':')) + '\n').encode())
            report[name]['semantic_sha256'] = semantic.hexdigest()
            if args.limit is None and args.seed == expected['seed'] and args.variants == expected['training_variants']:
                assert semantic.hexdigest() == expected['splits'][name]['semantic_sha256'], 'Generated episode content differs from released dataset'
        report.update(status="verified", elapsed_s=time.monotonic() - started)
        write_json(out / "stats.json", report); write_json(out / "READY.json", report)
        write_json(out / "generation_status.json", {**status, "status": "complete", "elapsed_s": time.monotonic() - started})
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        write_json(out / "generation_status.json", {**status, "status": "failed", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
