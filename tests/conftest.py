from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def env():
    from wordle_k64.environment import Engine
    from wordle_k64.sampler import ChallengeSampler
    e = Engine()
    return e, ChallengeSampler(e)


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    torch.set_num_threads(2)
    from tiny_model import create
    return create(str(tmp_path_factory.mktemp("tiny")))
