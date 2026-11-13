"""Trusted mechanics and a public-only decision boundary. Feedback uses BYG strings."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import random
import tempfile
import numpy as np
from .contracts import DATA, load_contract


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(canonical(parts).encode()).digest()[:16], "big")


def feedback(guess, answer):
    out, rest = ["B"] * 5, Counter()
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            out[i] = "G"
        else:
            rest[a] += 1
    for i, g in enumerate(guess):
        if out[i] != "G" and rest[g] > 0:
            out[i] = "Y"
            rest[g] -= 1
    return "".join(out)


def encode(pattern):
    return sum("BYG".index(c) * 3 ** i for i, c in enumerate(pattern))


def decode(code):
    return "".join("BYG"[(int(code) // 3 ** i) % 3] for i in range(5))


def pattern_block(guesses, answers):
    green = guesses[:, None, :] == answers[None, :, :]
    result = np.zeros(green.shape[:2], dtype=np.uint8)
    for i in range(5):
        available = ((answers[None, :, :] == guesses[:, None, i, None]) & ~green).sum(axis=2)
        used = np.zeros(green.shape[:2], dtype=np.uint8)
        for j in range(i):
            used += (guesses[:, None, i] == guesses[:, None, j]) & ~green[:, :, j]
        digit = np.where(green[:, :, i], 2, available > used).astype(np.uint8)
        result += digit * np.uint8(3 ** i)
    return result


@dataclass(frozen=True)
class State:
    answers: tuple
    history: tuple

    @property
    def remaining_turns(self):
        return 6 - len(self.history)


@dataclass(frozen=True)
class Decision:
    history: tuple
    words: tuple

    def __post_init__(self):
        if len(self.words) != 64 or len(set(self.words)) != 64:
            raise ValueError("Exactly 64 distinct candidate words are required.")
        if len(self.history) > 5:
            raise ValueError("No decision after the sixth guess.")
        if any(len(w) != 5 or not w.isascii() or not w.isalpha() or not w.islower()
               for w in self.words):
            raise ValueError("Candidate words must be five lowercase ASCII letters.")
        if any(len(g) != 5 or len(p) != 5 or set(p) - set("BYG") or p == "GGGGG"
               for g, p in self.history):
            raise ValueError("Invalid or terminal decision history.")

    def state_text(self):
        lines = ["LetterLock. Find a hidden five-letter word in at most six guesses.",
                 "G = correct letter and position. Y = an unmatched occurrence elsewhere.",
                 "B = no unmatched occurrence remains. Assign greens first, then yellows.",
                 "History:"]
        if not self.history:
            lines.append("(no guesses yet)")
        for turn, (word, pattern) in enumerate(self.history, 1):
            lines.append(f"Guess {turn}: " + " ".join(word.upper()))
            lines += [f"Position {i}: {c.upper()} -> {f}"
                      for i, (c, f) in enumerate(zip(word, pattern), 1)]
        lines.append(f"Attempts remaining: {6 - len(self.history)}.")
        return "\n".join(lines)

    def shuffled(self, rng):
        perm = rng.permutation(64)
        return Decision(self.history, tuple(self.words[i] for i in perm)), perm

    def to_dict(self):
        return {"history": [list(x) for x in self.history], "words": list(self.words)}

    @classmethod
    def from_dict(cls, value):
        return cls(tuple(tuple(x) for x in value["history"]), tuple(value["words"]))

    def drex_request(self):
        return {"state": self.state_text(), "questions": {"next_guess": {
            "type": "choice", "instructions": "Which word should be guessed next?",
            "criteria": {f"O{i:02d}": w.upper() for i, w in enumerate(self.words)}}}}


class Engine:
    def __init__(self, cache_dir=None):
        load_contract()
        self.answers = tuple((DATA / "answers.txt").read_text().split())
        self.guesses = tuple((DATA / "allowed.txt").read_text().split())
        self.gid = {w: i for i, w in enumerate(self.guesses)}
        self.aid = {w: i for i, w in enumerate(self.answers)}
        self.answer_to_guess = tuple(self.gid[w] for w in self.answers)
        self.dictionary_id = hashlib.sha256(canonical([self.answers, self.guesses]).encode()).hexdigest()
        cache = Path(cache_dir or os.environ.get("WORDLE_CACHE", ".cache/wordle"))
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"feedback-v1-{self.dictionary_id[:16]}.npy"
        checksum = path.with_suffix(".sha256")
        if not path.exists() or not checksum.exists():
            letters = lambda ws: np.frombuffer("".join(ws).encode(), dtype=np.uint8).reshape(-1, 5)
            guesses, answers = letters(self.guesses), letters(self.answers)
            matrix = np.empty((12972, 2315), dtype=np.uint8)
            for i in range(0, len(guesses), 64):
                matrix[i:i+64] = pattern_block(guesses[i:i+64], answers)
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as f:
                np.save(f, matrix, allow_pickle=False)
                temp = Path(f.name)
            digest = hashlib.sha256(temp.read_bytes()).hexdigest()
            os.replace(temp, path)
            checksum.write_text(digest)
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum.read_text().strip():
            raise ValueError("Feedback cache checksum mismatch; remove the cache and rebuild.")
        self.matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        assert self.matrix.shape == (12972, 2315) and self.matrix.dtype == np.uint8

    def state(self, history=()):
        history = tuple(tuple(x) for x in history)
        if len(history) > 6:
            raise ValueError("Too many guesses")
        keep = np.ones(2315, dtype=bool)
        for g, p in history:
            if g not in self.gid or len(p) != 5 or set(p) - set("BYG"):
                raise ValueError("Invalid public history")
            keep &= self.matrix[self.gid[g]] == encode(p)
        answers = tuple(int(a) for a in np.flatnonzero(keep))
        if not answers:
            raise ValueError("History contradicts all answers")
        return State(answers, history)

    def partitions(self, state, guess):
        buckets = {}
        for a in state.answers:
            buckets.setdefault(int(self.matrix[guess, a]), []).append(a)
        return buckets

    def entropy(self, state, guess):
        counts = np.bincount(self.matrix[guess, list(state.answers)], minlength=243)
        p = counts[counts > 0] / len(state.answers)
        return float(-(p * np.log2(p)).sum())


class Game:
    """Only this trusted object holds the actual secret; policy receives Decision only."""
    def __init__(self, engine, sampler, secret, seed):
        if secret not in engine.aid:
            raise ValueError("Unknown secret")
        self.engine, self.sampler, self._secret = engine, sampler, secret
        self.rng = random.Random(int(seed))
        self.history, self.done, self.won = (), False, False
        self._refresh()

    def _refresh(self):
        state = self.engine.state(self.history)
        self.options = list(self.sampler.sample(state))
        self.rng.shuffle(self.options)

    def decision(self):
        if self.done:
            raise ValueError("Game already ended")
        return Decision(self.history, tuple(self.engine.guesses[g] for g in self.options))

    def step(self, action):
        if self.done or not 0 <= int(action) < 64:
            raise ValueError("Illegal action")
        word = self.engine.guesses[self.options[int(action)]]
        pattern = feedback(word, self._secret)
        self.history += ((word, pattern),)
        self.won = pattern == "GGGGG"
        self.done = self.won or len(self.history) == 6
        if not self.done:
            self._refresh()
        return {"feedback": pattern, "done": self.done,
                "reward": int(self.won) if self.done else None}

    def terminal_reward(self):
        if not self.done:
            raise ValueError("Reward is defined only at game end")
        return float(self.won)
