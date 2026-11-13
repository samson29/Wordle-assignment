"""Complete resumable checkpoints: adapters, both heads, optimizer, RNG, contract."""
from pathlib import Path
import json
import os
import random
import shutil
import uuid
import numpy as np
import torch
from .contracts import fingerprint, require_fingerprint


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def capture_rng(rng):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "generator": rng.bit_generator.state, "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(saved, rng):
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    rng.bit_generator.state = saved["generator"]
    torch.set_rng_state(saved["torch"])
    if torch.cuda.is_available() and saved["cuda"] is not None:
        if len(saved["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same number of visible CUDA devices")
        torch.cuda.set_rng_state_all(saved["cuda"])


def save_checkpoint(policy, directory, *, optimizer=None, scheduler=None, rng=None,
                    progress=None, training_config=None):
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temp = directory.with_name(directory.name + ".writing-" + uuid.uuid4().hex[:8])
    temp.mkdir()
    adapters = ["policy"] + (["reference"] if policy.reference_head is not None else [])
    policy.lm.save_pretrained(temp / "adapters", selected_adapters=adapters)
    torch.save({k: v.detach().cpu() for k, v in policy.head.state_dict().items()}, temp / "head.pt")
    if policy.reference_head is not None:
        torch.save({k: v.detach().cpu() for k, v in policy.reference_head.state_dict().items()},
                   temp / "reference_head.pt")
    meta = {"format_version": 2, "contract": fingerprint(), "model_spec": policy.model_spec,
            "has_reference": policy.reference_head is not None, "progress": progress or {},
            "training_config": training_config or {}}
    (temp / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "rng": capture_rng(rng)}, temp / "trainer_state.pt")
    # Keep the previous complete checkpoint until the new directory is ready.
    backup = directory.with_name(directory.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    if directory.exists():
        os.replace(directory, backup)
    try:
        os.replace(temp, directory)
    except Exception:
        if backup.exists() and not directory.exists():
            os.replace(backup, directory)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    return directory


def resume_training(directory, optimizer, scheduler, rng, training_config):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    require_fingerprint(meta["contract"])
    if meta["training_config"] != training_config:
        raise ValueError("Resume training configuration differs; use --init for a new stage/run")
    # These are this application's own trusted optimizer/RNG files, not model weights.
    saved = torch.load(directory / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(saved["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(saved["scheduler"])
    restore_rng(saved["rng"], rng)
    return meta["progress"]


def metadata(directory):
    value = json.loads((Path(directory) / "metadata.json").read_text())
    require_fingerprint(value["contract"])
    return value
