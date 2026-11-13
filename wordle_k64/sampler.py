"""Port of the frozen explore-or-guess-k64-v1 membership policy; no secret input.

The legacy config's opening field is retained ONLY to preserve the policy hash
and history-seeded candidate membership. Game never forces an opening.
"""
from functools import lru_cache
import hashlib
import json
import random
import numpy as np
from .contracts import DATA, load_contract
from .environment import canonical, seed_for

def ranked_guesses(self, answers):
    """All guess IDs in ascending entropy order; stable dictionary-order ties."""
    feedback = np.asarray(self.engine.matrix[:, list(answers)], dtype=np.int32)
    offsets = np.arange(len(feedback), dtype=np.int32)[:, None] * 243
    counts = np.bincount((feedback + offsets).ravel(), minlength=len(feedback) * 243)
    counts = counts.reshape(len(feedback), 243)
    c_log_c = np.zeros(len(answers) + 1)
    nonzero = np.arange(1, len(c_log_c))
    c_log_c[1:] = nonzero * np.log2(nonzero)
    entropy = np.log2(len(answers)) - c_log_c[counts].sum(axis=1) / len(answers)
    return tuple((int(g) for g in np.argsort(np.round(entropy, 12), kind='stable')))

class WeakAnswerPolicy:
    """Choose an unhelpful exploitation action using only the public posterior.

Returns the exact number of secrets won by the resulting consistency-only
policy, but searches a bounded set of actions when more than two turns remain.
It is not a proof of the globally worst possible consistent-answer policy.
"""
    def __init__(self, engine, width=16):
        self.engine, self.width = engine, width
        self.solve = lru_cache(maxsize=200000)(self._solve)

    def _solve(self, answers, turns):
        n = len(answers)
        if not answers or turns <= 0:
            raise ValueError("Need a nonempty posterior and a positive turn budget")
        first = self.engine.answer_to_guess[answers[0]]
        if n <= turns or turns == 1:
            return min(n, turns), first
        guesses = np.array([self.engine.answer_to_guess[a] for a in answers])
        patterns = np.asarray(self.engine.matrix[guesses[:,None], np.array(answers)[None,:]])
        if turns == 2:
            buckets = 1 + (np.diff(np.sort(patterns,axis=1),axis=1)!=0).sum(axis=1)
            j = int(np.argmin(buckets))
            return int(buckets[j]), int(guesses[j])
        entropies, groups = [], []
        for row in patterns:
            codes, counts = np.unique(row,return_counts=True)
            p = counts / n
            entropies.append(round(float(-(p*np.log2(p)).sum()),12))
            groups.append(codes)
        order = np.argsort(entropies,kind="stable")[:self.width]
        best, chosen = n+1, None
        for j in order:
            wins = 1  # Guessing this surviving answer immediately wins one secret.
            for code in groups[j]:
                if code == 242:
                    continue
                child = tuple(a for a,c in zip(answers,patterns[j]) if c==code)
                wins += self.solve(child,turns-1)[0]
                if wins >= best:
                    break
            if wins < best:
                best, chosen = wins, int(guesses[j])
        return best, chosen

class ChallengeSampler:
    def __init__(self, engine):
        self.engine = engine
        self.config = json.loads((DATA/'candidate_policy.json').read_text())
        expected = {"version":"explore-or-guess-k64-root9-v3", "k":64, "consistent_slots":1,
                    "weak_search_width":16, "probe_slots":8, "distractor_slots":55}
        if any(self.config.get(k)!=v for k,v in expected.items()):
            raise ValueError("Update the versioned code and configuration together")
        self.policy_id = hashlib.sha256(canonical([self.config,engine.dictionary_id]).encode()).hexdigest()
        self.weak = WeakAnswerPolicy(engine,16)
        self.ranked_guesses = lru_cache(maxsize=128)(
            lambda answers: ranked_guesses(self,answers))
        self._membership = lru_cache(maxsize=20000)(self._build_membership)

    def _build_membership(self, state):
        if state.remaining_turns<=0 or (state.history and state.history[-1][1]=='GGGGG'):
            raise ValueError("Game has ended")
        if not state.history and state.answers == tuple(range(len(self.engine.answers))):
            # Frozen public root cache avoids repeating the expensive weak-answer search.
            # It is the same 64-word membership, not a forced first guess.
            root = json.loads((DATA / 'root_candidates.json').read_text())
            if root['policy_id'] != self.policy_id:
                raise ValueError('Root cache belongs to a different sampler')
            return tuple(self.engine.gid[w] for w in root['words'])
        _, weak = self.weak.solve(state.answers,state.remaining_turns)
        possible = {self.engine.answer_to_guess[a] for a in state.answers}
        excluded = possible | {self.engine.gid[g] for g,_ in state.history}
        ranked = [g for g in self.ranked_guesses(state.answers) if g not in excluded]
        selected = [weak] + ranked[-8:]
        excluded.update(selected)
        # Version the first-turn intervention without changing later membership.
        rng = random.Random(seed_for("membership",self.config["later_turn_policy_id"],state.history))
        while len(selected)<64:
            g = rng.randrange(len(self.engine.guesses))
            if g not in excluded:
                selected.append(g); excluded.add(g)
        return tuple(selected)

    def sample(self, state, rng=None):
        # Membership is deterministic; rng is intentionally unused. This also
        # makes every future sampler call inside planning exact, not Monte Carlo.
        return self._membership(state)

    def for_episode(self, state, *, seed, episode_id):
        options = list(self.sample(state,None))
        random.Random(seed_for("order",self.policy_id,seed,episode_id,state.history)).shuffle(options)
        return tuple(options)
