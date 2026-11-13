import json
import numpy as np
import pytest
from wordle_k64.contracts import load_contract, fingerprint, require_fingerprint
from wordle_k64.environment import Game, Decision, feedback, encode, decode
from wordle_k64.planner import ChallengePlanner
from wordle_k64.teacher import generate_episode, load_episodes


def test_split_is_locked_and_disjoint():
    s, e = load_contract()
    assert [len(s[k]) for k in ["train", "validation", "test"]] == [1565, 250, 500]
    assert not set(s["train"]) & (set(s["test"]) | set(s["validation"]))
    assert not set(s["validation"]) & set(s["test"])
    assert e["forced_opening"] is None
    with pytest.raises(ValueError):
        require_fingerprint({**fingerprint(), "split_sha256": "different"})


@pytest.mark.parametrize("guess,answer,pattern", [
    ("allee", "eagle", "YYBYG"), ("geese", "eerie", "BGYBG"),
    ("abbey", "kebab", "YYGYB"), ("crane", "plate", "BBGBG")])
def test_feedback(guess, answer, pattern):
    assert feedback(guess, answer) == pattern
    assert decode(encode(pattern)) == pattern


def test_sampler_public_only_and_root_cache(env):
    e, s = env
    a, b = Game(e, s, "crane", 42), Game(e, s, "jazzy", 42)
    assert a.decision() == b.decision()
    assert not a.history and len(a.options) == 64
    assert e.guesses[a.options[0]] == a.decision().words[0]
    for history in [(), (("crane", "BBGBG"),), (("crane", "BBGBG"), ("slate", "BGGGG"))]:
        state = e.state(history)
        c = s.sample(state)
        possible = {e.answer_to_guess[i] for i in state.answers}
        assert len(c) == len(set(c)) == 64
        assert len(set(c) & possible) == (1 if history else 0)
        assert not {e.gid[g] for g, _ in history} & set(c)


def test_game_reward_and_complete_teacher_trajectory(env, tmp_path):
    e, s = env
    split, _ = load_contract()
    game = Game(e, s, split["train"][0], 42)
    with pytest.raises(ValueError):
        game.terminal_reward()
    planner = ChallengePlanner(e, s)
    episode = generate_episode(e, s, planner, split["train"][0], "train", 0, 42)
    p = tmp_path / "episodes.jsonl"
    p.write_text(json.dumps(episode) + "\n")
    loaded = load_episodes(p, "train", e, s)
    assert loaded[0]["reward"] in [0, 1]
    assert 1 <= len(loaded[0]["steps"]) <= 6
    assert loaded[0]["steps"][0]["history"] == []
    with pytest.raises(ValueError):
        load_episodes(p, "validation")
    episode["steps"] = episode["steps"][:-1]
    p.write_text(json.dumps(episode) + "\n")
    with pytest.raises(ValueError):
        load_episodes(p, "train")


def test_matrix_scalar_agree(env):
    e, _ = env
    rng = np.random.default_rng(11)
    for g, a in zip(rng.integers(12972, size=200), rng.integers(2315, size=200)):
        assert decode(e.matrix[g, a]) == feedback(e.guesses[g], e.answers[a])
