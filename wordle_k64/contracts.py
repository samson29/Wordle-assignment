"""Immutable experiment definition. Never regenerate the split at training time."""
from pathlib import Path
import hashlib
import json

DATA = Path(__file__).resolve().parent / "data"
SPLIT_SHA256 = "3e61844360a38ac2c8517f1e679dd890793b5b3467b91513537ed1f76b554796"
ENV_SHA256 = "e4eed5e09a489e3844300f855ed2b2843df68953ca5c0b06aad6f56c921f121e"
ROOT_CACHE_SHA256 = "555ece6951470265b81f2a2d9c5d912493a8cc2c6e265c799729a514fd5833d0"
POLICY_SHA256 = "dd464557ac6fd21536f9a9470d367ea87223b2d07dad0380ed66a2def8abcd27"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_contract():
    for name, expected in [("split.json", SPLIT_SHA256), ("env.json", ENV_SHA256),
                           ("root_candidates.json", ROOT_CACHE_SHA256)]:
        if sha256(DATA / name) != expected:
            raise ValueError(f"Frozen {name} was changed. Restore the released manifest.")
    split = json.loads((DATA / "split.json").read_text())
    env = json.loads((DATA / "env.json").read_text())
    for name in ["answers.txt", "allowed.txt", "candidate_policy.json"]:
        expected = POLICY_SHA256 if name == "candidate_policy.json" else split[name + "_sha256"]
        if sha256(DATA / name) != expected:
            raise ValueError(f"Frozen dictionary/policy changed: {name}")
    answers = set((DATA / "answers.txt").read_text().split())
    groups = [set(split[k]) for k in ["train", "validation", "test"]]
    assert [len(g) for g in groups] == [1565, 250, 500]
    assert sum(len(g) for g in groups) == len(set.union(*groups)) == 2315
    assert set.union(*groups) == answers
    return split, env


def fingerprint():
    load_contract()
    return {"split_sha256": SPLIT_SHA256, "environment_sha256": ENV_SHA256,
            # Dataset contract covers either readout; checkpoint model_spec stores
            # the exact architecture and selects the corresponding loader.
            "readout": "character-or-contextual-pointer-v2", "candidate_policy_sha256": POLICY_SHA256}


def require_fingerprint(value):
    if value != fingerprint():
        raise ValueError("Checkpoint/data belongs to another split, environment, or readout.")
