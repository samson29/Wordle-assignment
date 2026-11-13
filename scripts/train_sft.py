"""Warm-start the shared head and LoRA on batches of COMPLETE teacher episodes."""
import argparse
import json
import math
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from peft import get_peft_model_state_dict
from _common import ROOT, model_args, build, environment, log, warmup, positive, memory_metrics
from wordle_k64.checkpoint import seed_everything, save_checkpoint, resume_training
from wordle_k64.contracts import sha256
from wordle_k64.teacher import load_episodes, flatten_episode_batch
from wordle_k64.environment import Decision
from wordle_k64.readout_diagnostics import load_panel, evaluate_panel, evaluate_order
from wordle_k64.sl_objectives import optimizer_groups, permutation_batch, align_log_probs, symmetric_kl
from wordle_k64.endgame_feedback import EndgameFeedback, fixed_panel_metrics
from wordle_k64.sft_validation import evaluate_games, write_json


def policy_digest(policy):
    state = get_peft_model_state_dict(policy.lm, adapter_name='policy')
    state = {**state, **{'head.'+k: v for k, v in policy.head.state_dict().items()}}
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    model_args(p)
    p.add_argument("--data", default="data/episodes/train.jsonl")
    p.add_argument("--out", default="runs/sl")
    p.add_argument("--resume")
    p.add_argument("--init", help="Start a separate SL branch from saved weights with a fresh optimizer")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--games-per-step", type=int, default=4)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, help="Pointer head LR; defaults to --lr")
    p.add_argument("--consistency-coef", type=float, default=0.0)
    p.add_argument("--consistency-seed", type=int, default=20261118)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-secrets", type=int, default=250)
    p.add_argument("--eval-batch", type=int, default=8)
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[1001])
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--diagnostic-data", help="Validation teacher episodes for a fixed public-state panel")
    p.add_argument("--turn5-panel", default=str(ROOT / "data/sft_turn5_panel.json"), help="Frozen validation-only hard turn-five states")
    p.add_argument("--panel-limit", type=int, help="Smoke tests only")
    p.add_argument("--diagnostic-limit", type=int, help="Smoke tests only")
    p.add_argument("--expected-head-sha256", help="Require the character control's identical initial head")
    p.add_argument('--config', type=Path, default=ROOT/'configs/sft.json')
    p.add_argument('--experiment', choices=['PSL-head', 'PSL-head-consloss', 'LM-head'], default='PSL-head-consloss')
    preliminary, _ = p.parse_known_args()
    recipes = json.loads(preliminary.config.read_text())
    p.set_defaults(**recipes['common'], **recipes['experiments'][preliminary.experiment])
    args = p.parse_args()
    if args.head_lr is None:
        args.head_lr = args.lr
    if args.init and args.resume:
        p.error("Use --init OR --resume")
    positive(epochs=args.epochs, games_per_step=args.games_per_step, micro_batch=args.micro_batch,
             eval_every=args.eval_every, save_every=args.save_every, lr=args.lr, head_lr=args.head_lr)
    if args.max_steps is not None:
        positive(max_steps=args.max_steps)
    if not math.isfinite(args.consistency_coef) or args.consistency_coef < 0:
        p.error('--consistency-coef must be finite and nonnegative')
    if not 1 <= args.eval_secrets <= 250:
        p.error("--eval-secrets must be 1..250 (use 250 for final selection)")
    rng = seed_everything(args.seed)
    consistency_rng = np.random.default_rng(args.consistency_seed)
    split, engine, sampler = environment()
    feedback = EndgameFeedback(engine)
    turn5_document = json.loads(Path(args.turn5_panel).read_text())
    if turn5_document.get('source_split') != 'validation':
        p.error('Turn-five panel must come exclusively from validation')
    turn5_panel = turn5_document['states'][:args.panel_limit]
    assert turn5_panel
    episodes = load_episodes(args.data, "train", engine, sampler)
    policy = build(args, args.resume or args.init)
    params = policy.trainable_parameters()
    opt = torch.optim.AdamW(optimizer_groups(policy, args.lr, args.head_lr), weight_decay=0)
    sched = warmup(opt, args.warmup)
    config = {"stage": "sl", "data_sha256": sha256(args.data), "games_per_step": args.games_per_step,
              "lr": args.lr, "warmup": args.warmup, "micro_batch": args.micro_batch,
              "eval_secrets": args.eval_secrets, "eval_seeds": args.eval_seeds}
    config['architecture'] = policy.model_spec['architecture']
    config['target_rule'] = 'public-singleton-and-offered-v1'
    config.update(head_lr=args.head_lr, consistency_coef=args.consistency_coef,
                  consistency_seed=args.consistency_seed,
                  consistency_objective='word-aligned-symmetric-kl-v1',
                  optimizer_groups=['backbone_lora', 'pointer_head'])
    config['diagnostic_data_sha256'] = sha256(args.diagnostic_data) if args.diagnostic_data else None
    config.update(turn5_panel_sha256=sha256(args.turn5_panel), eval_batch=args.eval_batch,
                  eval_every=args.eval_every, seed=args.seed, panel_limit=args.panel_limit,
                  diagnostic_limit=args.diagnostic_limit,
                  validation_checkpoint_policy='retain_every_validation_including_step_zero',
                  turn5_diagnostics='public_two_turn_solver_values_fixed_and_live',
                  validation_protocol='full_games_and_fixed_order_panels_v1')
    panel = load_panel(args.diagnostic_data)[:args.diagnostic_limit] if args.diagnostic_data else None
    if args.init:
        config["init_checkpoint"] = str(Path(args.init).resolve())
    elif args.resume:
        previous = json.loads((Path(args.resume) / "metadata.json").read_text())["training_config"]
        if "init_checkpoint" in previous:
            config["init_checkpoint"] = previous["init_checkpoint"]
    progress = {"step": 0, "best_validation": -1, "order": rng.permutation(len(episodes)).tolist(), "cursor": 0}
    if args.resume:
        progress = resume_training(args.resume, opt, sched, rng, config)
        consistency_rng.bit_generator.state = progress['consistency_rng_state']
    elif (Path(args.out) / "last").exists():
        p.error("Run already exists; use --resume or a new --out")
    total = args.max_steps or args.epochs * math.ceil(len(episodes) / args.games_per_step)
    write_json(Path(args.out) / 'configuration.json', {**config, 'model_spec': policy.model_spec,
               'requested_total_steps': total, 'training_games': len(episodes),
               'data': str(Path(args.data).resolve()), 'initialization': args.init or 'fresh_base_model'})

    def save(name):
        progress['consistency_rng_state'] = consistency_rng.bit_generator.state
        save_checkpoint(policy, Path(args.out) / name, optimizer=opt, scheduler=sched,
                        rng=rng, progress=progress, training_config=config)

    opening = Decision((), tuple(engine.guesses[g] for g in sampler.sample(engine.state(()))))

    def evaluate():
        eval_started = time.monotonic()
        log(args.out, stage='validation_started', step=progress['step'])
        result, records = evaluate_games(policy, engine, sampler, split["validation"][:args.eval_secrets],
                                        args.eval_seeds, args.eval_batch, feedback)
        order_records = {}
        if panel is not None:
            result['diagnostics'] = evaluate_panel(policy, panel, args.eval_batch, include_order_rows=True)
            order_records['general_panel'] = result['diagnostics'].pop('order_rows')
        result['fixed_turn5_panel'] = fixed_panel_metrics(policy, turn5_panel, feedback, args.eval_batch)
        turn5_order, _, turn5_rows = evaluate_order(policy, [Decision.from_dict(s) for s in turn5_panel], args.eval_batch)
        result['turn5_order'] = turn5_order
        order_records['hard_turn5_panel'] = turn5_rows
        probs = policy.score([opening], batch_size=1)[0].exp()
        opening_metrics = {"word": opening.words[int(probs.argmax())],
                           "salet_probability": float(probs[opening.words.index("salet")]),
                           "entropy": float(-(probs * probs.clamp_min(1e-30).log()).sum())}
        path = Path(args.out) / "validation" / f"step_{progress['step']:05d}.json"
        checkpoint_name = f"checkpoints/step_{progress['step']:05d}"
        improved = result['win_rate'] > progress['best_validation']
        if improved:
            progress['best_validation'] = result['win_rate']
        # Preserve every evaluated checkpoint, including optimizer, scheduler and RNG.
        save(checkpoint_name)
        if improved:
            save('best')
        save('last')
        result.update(checkpoint=checkpoint_name, policy_sha256=policy_digest(policy))
        write_json(path, {"step": progress["step"], "opening": opening_metrics, **result})
        write_json(Path(args.out) / 'validation_games' / path.name, {'split': 'validation', 'games': records})
        write_json(Path(args.out) / 'order_predictions' / path.name, order_records)
        compact = {k: v for k, v in result.items() if k != 'per_answer_success'}
        compact['fixed_turn5_panel'] = {k: v for k, v in result['fixed_turn5_panel'].items() if k != 'decisions'}
        log(args.out, stage="validation", step=progress["step"], opening=opening_metrics,
            evaluation_s=time.monotonic() - eval_started, memory=memory_metrics(),
            **compact)

    if not args.resume:
        head_hash = hashlib.sha256()
        for name, tensor in policy.head.state_dict().items():
            head_hash.update(name.encode()); head_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        if args.expected_head_sha256 and head_hash.hexdigest() != args.expected_head_sha256:
            raise AssertionError('Head initialization differs from the ongoing control')
        log(args.out, stage="initialization", step=0, rank=policy.model_spec["rank"],
            alpha=policy.model_spec["alpha"], head_seed=args.head_seed,
            initial_head_sha256=head_hash.hexdigest(),
            initial_policy_sha256=policy_digest(policy),
            trainable_parameters=sum(p.numel() for p in params), training_games=len(episodes),
            architecture=policy.model_spec['architecture'], target_rule=config['target_rule'],
            optimizer_groups=[{'name': g['name'], 'peak_lr': g['initial_lr'],
                               'parameters': sum(p.numel() for p in g['params'])} for g in opt.param_groups],
            consistency_coef=args.consistency_coef, consistency_seed=args.consistency_seed)
        evaluate()
        save("last")

    for step in range(progress["step"] + 1, total + 1):
        step_started = time.monotonic()
        learning_rate = opt.param_groups[0]["lr"]
        head_learning_rate = opt.param_groups[1]['lr']
        if progress["cursor"] >= len(episodes):
            progress["order"] = rng.permutation(len(episodes)).tolist()
            progress["cursor"] = 0
        start = progress["cursor"]
        selected = progress["order"][start:start + args.games_per_step]
        progress["cursor"] += len(selected)
        decisions, masks = flatten_episode_batch([episodes[i] for i in selected], rng)
        policy.train()
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        supervised_loss = consistency_loss = 0.0
        consistency_flips = consistency_tv = 0.0
        consistency_decisions = 0
        correct = 0
        accepted_mass = 0.0
        entropy_sum = 0.0
        turn_sums = {}
        closing_flags = [s.get('closing_target') is not None for j in selected for s in episodes[j]['steps']]
        closing_correct = closing_count = 0
        for i in range(0, len(decisions), args.micro_batch):
            log_p = policy.log_probs(decisions[i:i + args.micro_batch])
            mask = torch.as_tensor(masks[i:i + args.micro_batch], device=log_p.device)
            row_losses = -torch.logsumexp(log_p.masked_fill(~mask, -torch.inf), -1)
            supervised = row_losses.sum() / len(decisions)
            consistency = log_p.new_zeros(())
            if args.consistency_coef:
                shuffled, inverse = permutation_batch(decisions[i:i + args.micro_batch], consistency_rng)
                log_q = align_log_probs(policy.log_probs(shuffled), inverse)
                consistency = symmetric_kl(log_p, log_q).sum() / len(decisions)
                with torch.no_grad():
                    consistency_flips += int((log_p.argmax(-1) != log_q.argmax(-1)).sum())
                    consistency_tv += float((log_p.exp() - log_q.exp()).abs().sum() * 0.5)
                    consistency_decisions += len(shuffled)
            loss = supervised + args.consistency_coef * consistency
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite SL loss')
            total_loss += float(loss.detach())
            supervised_loss += float(supervised.detach())
            consistency_loss += float(consistency.detach())
            with torch.no_grad():
                correct += int(mask.gather(1, log_p.argmax(-1, keepdim=True)).sum())
                accepted_mass += float((log_p.exp() * mask).sum())
                entropy_sum += float(-(log_p.exp() * log_p).sum())
                for j, value in enumerate(row_losses.detach().cpu().tolist()):
                    turn = str(len(decisions[i+j].history)+1)
                    row = turn_sums.setdefault(turn, {'decisions': 0, 'loss_sum': 0.0, 'correct': 0})
                    hit = int(mask[j, log_p[j].argmax()])
                    row['decisions'] += 1; row['loss_sum'] += value; row['correct'] += hit
                    if closing_flags[i+j]:
                        closing_count += 1; closing_correct += hit
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite gradients")
        opt.step()
        sched.step()
        progress["step"] = step
        step_s = time.monotonic() - step_started
        by_turn = {t: {'decisions': r['decisions'], 'loss': r['loss_sum']/r['decisions'],
                       'accepted_accuracy': r['correct']/r['decisions']} for t, r in turn_sums.items()}
        log(args.out, stage="sl", step=step, loss=total_loss,
            supervised_loss=supervised_loss, consistency_loss=consistency_loss,
            weighted_consistency_loss=args.consistency_coef * consistency_loss,
            consistency_coef=args.consistency_coef, consistency_decisions=consistency_decisions,
            train_order_flip_rate=consistency_flips/consistency_decisions if consistency_decisions else None,
            train_order_total_variation=consistency_tv/consistency_decisions if consistency_decisions else None,
            complete_games=len(selected), decisions=len(decisions),
            teacher_action_accuracy=correct / len(decisions),
            accepted_probability=accepted_mass / len(decisions),
            entropy=entropy_sum / len(decisions), grad_norm=float(norm),
            learning_rate=learning_rate, next_learning_rate=opt.param_groups[0]["lr"],
            backbone_learning_rate=learning_rate, head_learning_rate=head_learning_rate,
            next_head_learning_rate=opt.param_groups[1]['lr'],
            step_s=step_s, decisions_per_s=len(decisions) / step_s,
            epoch=(step - 1) // math.ceil(len(episodes) / args.games_per_step) + 1,
            total_steps=total, memory=memory_metrics(), by_turn=by_turn,
            closing_decisions=closing_count,
            closing_accuracy=closing_correct/closing_count if closing_count else None,
            longest_prompt_tokens=policy.longest_prompt)
        if step % args.eval_every == 0 or step == total:
            evaluate()
        if step % args.save_every == 0 or step == total:
            save("last")


if __name__ == "__main__":
    main()
