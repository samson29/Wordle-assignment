"""Teacher: exact policy improvement over a deterministic continuation.

A returned value of one certifies a guaranteed win. Values below one describe
that continuation, not a proof of globally maximal success. With <=2 turns all
64 actions are evaluated; earlier decisions stop at the first guarantee.
"""
from functools import lru_cache
import random
import numpy as np
from .environment import State, decode

class RestrictedSolver:
    """Evaluate a fixed continuation policy using public hypothetical feedback.

    For each offered action, integrate feedback over ALL remaining answers.
    Candidate membership is deterministic. The
    result is Q of that continuation policy, not necessarily globally optimal Q.
    One/two-turn action values and guaranteed small states are evaluated exactly.
    """
    def __init__(self, engine, sampler, *, rollouts=8, seed=17):
        if rollouts < 1:
            raise ValueError("rollouts must be positive")
        self.engine, self.sampler = engine, sampler
        self.rollouts, self.seed = rollouts, seed
        self._cached_value = lru_cache(maxsize=50000)(self._uncached_value)

    def _possible(self, state):
        return {self.engine.answer_to_guess[a] for a in state.answers}

    def _base_choice(self, state, candidates):
        possible = self._possible(state)
        consistent = [g for g in candidates if g in possible]
        n, t = len(state.answers), state.remaining_turns
        if t == 1:
            return min(consistent or candidates, key=lambda g: self.engine.guesses[g])
        if n <= t:
            # At least one consistent answer is offered: each miss eliminates an answer.
            choices = consistent
        else:
            choices = candidates
        if t == 2:
            return min(choices, key=lambda g: (-len(self.engine.partitions(state, g)),
                       -(g in possible), self.engine.guesses[g]))
        return min(choices, key=lambda g: (-round(self.engine.entropy(state, g), 12),
                   -(g in possible), self.engine.guesses[g]))

    def _value(self, state, rng):
        return self._cached_value(state)

    def _uncached_value(self, state):
        rng = random.Random(0)
        """Expected success before drawing the next candidate set."""
        n, t = len(state.answers), state.remaining_turns
        if t <= 0:
            return 0.0
        if n <= t:  # at least one consistent answer is offered at every subsequent turn.
            return 1.0
        if t == 1:
            # At least one possible answer always offered, uniform secret prior.
            return 1.0 / n
        candidates = self.sampler.sample(state, rng)
        guess = self._base_choice(state, candidates)
        return self._action_value(state, guess, rng)

    def _action_value(self, state, guess, rng):
        n, t = len(state.answers), state.remaining_turns
        if t <= 0:
            return 0.0
        if t == 1:
            return float(guess in self._possible(state)) / n
        groups = self.engine.partitions(state, guess)
        if t == 2:
            # Each nonempty feedback bucket yields exactly one success in expectation
            # on the last turn; GGGGG is an immediate success. Future options always
            # contain >=1 consistent answer, sampled independently of the secret.
            return len(groups) / n
        total = 0.0
        for pattern, answers in sorted(groups.items()):
            if pattern == 242:
                total += len(answers)
            else:
                child = State(tuple(answers), state.history + ((self.engine.guesses[guess], decode(pattern)),))
                total += len(answers) * self._value(child, rng)
        return total / n



class ChallengePlanner(RestrictedSolver):
    """Exact policy improvement over a fixed continuation, not global DP.

Future option membership is deterministic. _action_value integrates all possible
feedback; no future-set sampling error remains. Once a value of 1 is reached in
tie-break order, no later action can improve it. This avoids evaluating many
irrelevant distractors without changing the decision.
"""
    def __init__(self, engine, sampler):
        super().__init__(engine,sampler,rollouts=1)

    def choose(self, state, candidates):
        if state.remaining_turns<=0 or (state.history and state.history[-1][1]=='GGGGG'):
            raise ValueError("Game has ended")
        candidates = tuple(candidates)
        if len(candidates)!=64 or len(set(candidates))!=64:
            raise ValueError("Expected 64 unique offered words")
        possible = self._possible(state)
        order = sorted(candidates,key=lambda g:(-(g in possible),
                         -round(self.engine.entropy(state,g),12),self.engine.guesses[g]))
        chosen, best, rows = None, -1.0, []
        for g in order:
            value = self._action_value(state,g,random.Random(0))
            rows.append({"guess":self.engine.guesses[g],"guess_id":g,
                         "continuation_win_probability":value,
                         "value_exact_for_continuation":True})
            if value > best+1e-12:
                chosen, best = g, value
            if value >= 1-1e-12 and state.remaining_turns > 2:
                break
        return chosen, rows
