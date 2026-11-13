# Wordle learner

This repo includes my experiments with SFT+RL for the assignment. Code was fully written by GPT-6 Astra. 

## Files

| Path | Contents |
|---|---|
| `wordle_k64/` | Environment, frozen word lists/split, candidate builder, readouts, objectives and diagnostics |
| `scripts/` | Data generation, SFT, RL, train-only mining, evaluation and result export |
| `configs/` | Three SFT recipes and two RL recipes |
| `data/` | Small frozen validation panels, mined training-answer pool and one public training state for gradient checks |
| `results/` | Training/validation CSV logs, held-out metrics, three failure examples and source hashes |
| `checkpoints/` | Selected adapters, heads, optimizer/RNG states and checkpoint metadata |
| `tests/` | CPU mechanics tests with tiny models and small complete-game fixtures |


## Running

Run commands from this repository root. The tested environment uses Python 3.11, PyTorch 2.10.0 with CUDA 12.8, Transformers 5.22.2 and PEFT 0.21.2; dependency versions are in `pyproject.toml`. Existing prepared environments can run the scripts directly. For a new compatible environment, install with `pip install -e '.[test]'` after installing the appropriate CUDA build of PyTorch.

The backbone is `Qwen/Qwen3.6-35B-A3B`; the original cached model revision was `345d467b117d1fd5eda9b879a5988de9ae5f47a7`. Use that snapshot for reproduction. `--model` accepts a local snapshot directory. Model loading is local-only by default; `--allow-download` enables downloads. Real training uses one process across two visible GPUs, without CPU offload. The reported experiments used A100 80 GB GPUs. `WORDLE_CACHE` and `HF_HOME` can point to existing caches.

The recorded GPU environment also used Flash Linear Attention 0.6.0, causal-conv1d 1.7.0 and Triton 3.6.0. After installing CUDA-compatible PyTorch and a matching CUDA build toolkit, install these with `pip install --no-build-isolation -e '.[gpu]'`. They are optional for CPU mechanics tests. The default model revision is pinned in the loader; add `--allow-download` on the first GPU run if that snapshot is not cached.

```bash
# Generate complete train/validation teacher games; test answers are excluded.
python scripts/make_data.py --out data/episodes --variants 10 --workers 4

# SFT recipes: PSL-head, PSL-head-consloss, LM-head. Losses and metrics are logged.
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_sft.py \
  --experiment PSL-head-consloss --out runs/PSL-head-consloss

# Complete-game evaluation, all-turn order checks and public endgame diagnostics.
CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate.py --policy checkpoint \
  --checkpoint checkpoints/PSL-head-consloss/step_01100 \
  --split test --out runs/eval-cons1100

# The four CPU baselines: random, heuristic-random, heuristic-distinct, heuristic-salet.
CUDA_VISIBLE_DEVICES='' python scripts/evaluate.py --policy heuristic-salet \
  --split test --out runs/heuristic-salet

# Untrained native LM-head control
CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate.py --policy zero-shot \
  --split test --out runs/zero-shot

# RL: choose RL-A or RL-B. The bundled pool matches the included starting checkpoint.
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_rl.py --arm RL-A \
  --init checkpoints/PSL-head-consloss/step_01100 --out runs/RL-A

# Evaluate the included RL checkpoint.
CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate.py --policy checkpoint \
  --checkpoint checkpoints/RL-A/iteration_00012 \
  --split test --out runs/eval-RL-A

# CPU mechanics tests; no model downloads or GPUs.
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

For a newly trained checkpoint, mine a matching pool before RL: run `scripts/mine.py --checkpoint PATH --out runs/mining --shard N` for each N = 0, 1, 2, 3, setting a GPU pair for each process. Then run `python scripts/freeze_pool.py --out runs/mining` and pass `--hard-pool runs/mining/hard_train_secrets.json` to the RL trainer. The trainer verifies pool/checkpoint identity and train-split membership. Runs created by these scripts can resume with the same arguments plus `--resume RUN/last`; RL requires the same `--init` argument too. Included archived checkpoints retain their original training metadata; their settings are not interchangeable with the renamed recipes for strict optimizer resume.

The generator combines the original generation and closing-relabel steps. Episode content is preserved, but JSON serialization can differ from archived byte hashes. Selected checkpoints are included locally; the frozen Qwen backbone is loaded from its separate model cache. Fresh retraining is not a claim of bit-identical reproduction across hardware/software.

## Reproducing the numbers

The repository includes a compressed observation file, `results/heldout_samples.jsonl.gz`, containing all 13,500 held-out games across nine policies. It retains guess/feedback trajectories, selected-option confidence, solver labels and the individual order-comparison statistics. It is sufficient to recompute the main result tables without loading Qwen. Dense per-option logits are regenerated by `evaluate.py`.

Arithmetic-only verification needs Python and `numpy==2.4.6`; environment replay and training tests use the full installation above.

```bash
# Recount wins, turn histograms, conditional mean guesses, ECE, flips, TV,
# endgame rates and 95% answer-cluster bootstrap CIs; check exported tables.
CUDA_VISIBLE_DEVICES='' python scripts/verify_results.py

# Also replay every candidate action and feedback, check public endgame labels,
# and verify all selected checkpoint files against their SHA-256 manifest.
CUDA_VISIBLE_DEVICES='' python scripts/verify_results.py --replay --checkpoints

# After any learned-policy evaluation, calculate detailed calibration bins.
python scripts/calibration.py --evaluation runs/eval-cons1100 \
  --output-prefix runs/eval-cons1100/calibration

# Redraw the released overall reliability diagrams and turn-of-win histograms.
# Install plotting dependencies first: pip install -e '.[plot]'
python scripts/plot_overall_calibration.py
python scripts/plot_test_turns.py
```

The verification uses 500 answer clusters, keeping all three seeds and all turns of each answer together, with 2,000 bootstrap draws and seed 722. Mean guesses is conditional on winning. Order metrics compare the original distribution with reverse order and two random permutations, aligned by word, over every played decision; the same-order rerun is a separate control. ECE uses raw top-option probabilities and 15 equal-width bins. A correct choice has maximal public solver continuation value, with an immediate-closing requirement when one offered answer remains. Turns 5–6 have exact values; earlier turns use a fixed continuation. Each policy is measured on its own visited states.

The compact verifier checks the arithmetic from recorded predictions; reproducing the model predictions themselves requires the GPU evaluation commands above. `--replay` independently verifies game legality, feedback, posterior sizes, closing labels and turn-five values. It does not rerun the earlier-turn solver search. The 500-answer test set was used for exploratory failure analysis during development; it is not a new blind evaluation.

`make_data.py` verifies complete trajectories and split membership before writing `READY.json`. With the default seed and ten variants, it also checks the semantic hashes in `data/teacher_manifest.json`: 15,650 training games / 59,823 decisions and 250 validation games / 1,079 decisions. The complete regenerated dataset was compared field-for-field with the training archive. Add `--max-steps 1100` to a fresh SFT command to stop at the selected checkpoint stage.

`export_results.py --archive PATH` is an optional maintainer tool for importing the original experiment archive. It is not needed to prepare data, train, evaluate, verify the released results or redraw the figures.

## Checkpoints on GitHub

The five selected checkpoints occupy approximately 1.1 GiB, including optimizer/RNG state. `.gitattributes` assigns checkpoint tensors to Git LFS; checkpoint metadata and the integrity manifest use ordinary Git. Install Git LFS before adding the repository files:

```bash
# In the new Git repository, before git add:
git lfs install --local
git add .
git lfs ls-files

# After cloning a published repository:
git lfs pull
python scripts/verify_results.py --checkpoints
```

A plain download of Git LFS pointer files is insufficient for inference. The Qwen backbone is separate and is not bundled. The report source/template is excluded; figures and the scripts/data that generate them are included.
