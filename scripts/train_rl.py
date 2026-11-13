"""Complete-game RL with a hard-answer curriculum (RL-A) and an additional low-information penalty (RL-B)."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from _common import ROOT, model_args, build, environment, log, warmup, memory_metrics
from wordle_k64.checkpoint import seed_everything, save_checkpoint, resume_training
from wordle_k64.environment import Decision, seed_for
from wordle_k64.sl_objectives import optimizer_groups, permutation_batch, align_log_probs, symmetric_kl
from wordle_k64.endgame_feedback import whole_game_minibatches, trajectory_metrics
from wordle_k64.probe_study import (ProbeFeedback, digest, write_json, play_complete, game_specs,
    records_from_games, secret_schedule, rloo_advantages, loss_components)
from wordle_k64.probe_validation import evaluate_complete, attach_penalty_metrics
from wordle_k64.low_information import LowInformationPenalty, summarize_penalties


def preflight(policy, feedback, hard_record, out):
    """No optimizer update: verify real-model ratios, probe gradients and frozen anchor."""
    d=Decision.from_dict(hard_record);desc=feedback.describe(d)
    assert desc['probe_eligible']
    initial=digest(policy);ref_digest=digest(policy,True)
    with policy.reference():reference=policy.score([d],1)
    old=policy.score([d],1);old_sample=(old/1.2).log_softmax(-1)
    policy.train();policy.zero_grad(set_to_none=True)
    raw=policy.log_probs([d],1.);sample=(raw/1.2).log_softmax(-1)
    delta=float((raw.detach().cpu()-old).abs().max())
    sample_delta=float((sample.detach().cpu()-old_sample).abs().max())
    assert max(delta,sample_delta)<.02, f'Unchanged-policy log probabilities differ: {delta}, {sample_delta}'
    mask=torch.as_tensor(desc['optimal_mask'],device=raw.device)
    loss=-torch.logsumexp(raw[0,mask],0)
    loss.backward()
    norms={'head':sum(float(p.grad.detach().float().square().sum()) for p in policy.head.parameters() if p.grad is not None),
           'adapter':sum(float(p.grad.detach().float().square().sum()) for n,p in policy.named_parameters() if '.policy.' in n and p.grad is not None)}
    assert all(np.isfinite(x) and x>0 for x in norms.values()), norms
    assert all(p.grad is None for n,p in policy.named_parameters() if '.reference.' in n or n.startswith('reference_head.'))
    assert digest(policy)==initial and digest(policy,True)==ref_digest
    policy.zero_grad(set_to_none=True)
    result={'status':'passed','optimizer_updates':0,'max_full_logp_delta':delta,
        'max_tempered_logp_delta':sample_delta,'head_gradient_squared_norm':norms['head'],
        'adapter_gradient_squared_norm':norms['adapter'],'probe_loss':float(loss.detach()),
        'reference_equals_initial':initial==ref_digest,'reference_unchanged':True,
        'raw_policy_reference_max_delta':float((old-reference).abs().max()),
        'turn':5,'remaining_answers':desc['n'],'memory':memory_metrics()}
    write_json(out/'gpu_preflight.json',result)
    return result


def main():
    p=argparse.ArgumentParser();model_args(p)
    p.add_argument('--config',type=Path,default=ROOT/'configs/rl.json')
    p.add_argument('--arm',choices=['RL-A','RL-B'],required=True)
    p.add_argument('--init',type=Path,required=True,help='Starting SFT checkpoint used to mine the hard pool')
    p.add_argument('--hard-pool',type=Path,default=ROOT/'data/hard_train_secrets.json')
    p.add_argument('--turn5-panel',type=Path,default=ROOT/'data/rl_turn5_panel.json')
    p.add_argument('--preflight-state',type=Path,default=ROOT/'data/preflight_train_state.json')
    p.add_argument('--diagnostic-data',type=Path,default=Path('data/episodes/validation.jsonl'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--stop-after',type=int);p.add_argument('--skip-initial-evaluation',action='store_true')
    args=p.parse_args()
    recipes=json.loads(args.config.read_text())
    plan={**recipes['common'],**recipes['experiments'][args.arm]}
    out=args.out
    assert not (out/'last').exists() or args.resume, 'Refuse duplicate fresh run'
    args.seed=plan['seed'];args.architecture='pointer'
    rng=seed_everything(args.seed);split,engine,sampler=environment();feedback=ProbeFeedback(engine)
    coefficient=plan['reward']['low_information_penalty_per_turn']
    penalty=LowInformationPenalty(engine,coefficient) if coefficient else None
    assert not set(split['train']) & set(split['validation']+split['test'])
    pool_path=args.hard_pool
    pool=json.loads(pool_path.read_text());assert pool['status']=='frozen'
    hard=pool['secrets'];assert set(hard)<=set(split['train'])
    assert len(hard)>=16
    config=copy.deepcopy(plan)
    config.update(status='IMPLEMENTED',stage='complete_game_probe_rl_v1',smoke=args.smoke,
        hard_pool_sha256=hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        fixed_validation_panel_sha256=hashlib.sha256(args.turn5_panel.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [*sorted((ROOT/'wordle_k64').glob('*.py')),
                                 ROOT/'scripts/train_rl.py', ROOT/'scripts/_common.py']})
    if args.smoke:
        config.update(iterations=2,secrets_per_iteration=2,games_per_secret=2,games_per_iteration=4,
            games_per_optimizer_update=4,optimizer_updates_per_iteration=1,optimizer_updates_total=2,new_training_games=8)
        config['validation'].update(iterations=[0,1,2],secrets=2,seeds=[1001],games=2)
    write_json(out/'configuration.json',config)
    policy=build(args,args.resume or args.init)
    if not args.resume:policy.anchor_reference()
    params=policy.trainable_parameters();settings=config['optimizer']
    opt=torch.optim.AdamW(optimizer_groups(policy,settings['lora_lr'],settings['head_lr']),
        weight_decay=settings['weight_decay'],betas=tuple(settings['betas']),eps=settings['eps'])
    scheduler=warmup(opt,settings['warmup_updates'])
    progress={'iteration':0,'optimizer_step':0,'best_selection':None,
              'reference_sha256':digest(policy,True),'completed_games':0,'train_secrets_seen':[],
              'pending_validation':False}
    if args.resume:progress=resume_training(args.resume,opt,scheduler,rng,config)
    assert digest(policy,True)==progress['reference_sha256']
    if not args.resume:
        assert digest(policy)==progress['reference_sha256']==pool['policy_sha256']
    start_time=time.monotonic()
    def event(stage,**kw):
        log(out,stage=stage,step=progress['optimizer_step'],iteration=progress['iteration'],
            session_elapsed_s=time.monotonic()-start_time,**kw)
    event('initialization',arm=plan['arm'],model_spec=policy.model_spec,
          policy_sha256=digest(policy),reference_sha256=progress['reference_sha256'],
          trainable_parameters=sum(x.numel() for x in params),memory=memory_metrics())
    def save(name):
        return save_checkpoint(policy,out/name,optimizer=opt,scheduler=scheduler,rng=rng,
                               progress=progress,training_config=config)
    panel=json.loads(args.turn5_panel.read_text())['states']
    diagnostics=args.diagnostic_data
    if args.smoke:panel=panel[:2];diagnostics=None
    def evaluate():
        iteration=progress['iteration'];save(f'checkpoints/iteration_{iteration:05d}')
        event('validation_started',games=config['validation']['games'])
        destination=out/f'validation/iteration_{iteration:05d}'
        report=evaluate_complete(policy,engine,sampler,split['validation'][:config['validation']['secrets']],
            config['validation']['seeds'],feedback,panel,destination,event,
            diagnostic_data=diagnostics,penalty_coefficient=coefficient or None)
        score=[report['wins'],-report['fixed_turn5_panel']['mean_solver_regret'],
               -report['order']['flip_rate']['mean'],-iteration]
        if progress['best_selection'] is None or tuple(score)>tuple(progress['best_selection']):
            progress['best_selection']=score
            write_json(out/'best_checkpoint.json',{'iteration':iteration,'selection_key':score,
                'checkpoint':str((out/f'checkpoints/iteration_{iteration:05d}').resolve())})
        compact=copy.deepcopy(report);compact['fixed_turn5_panel'].pop('decisions',None)
        event('validation',**compact)
        progress['pending_validation']=False
        return report
    if not args.resume:
        hard_record=json.loads(args.preflight_state.read_text())
        event('gpu_preflight',**preflight(policy,feedback,hard_record,out))
        progress['pending_validation']=not args.skip_initial_evaluation
        save('last')
        if progress['pending_validation']:evaluate()
        save('last')
    elif progress.get('pending_validation'):
        evaluate();save('last')
    limit=min(config['iterations'],args.stop_after or config['iterations'])
    for iteration in range(progress['iteration']+1,limit+1):
        iteration_start=time.monotonic()
        sampled,sources=secret_schedule(split['train'],hard,config['curriculum']['hard_secret_fraction'],args.seed,iteration)
        if args.smoke:sampled=[sampled[0],hard[0]];sources=['uniform','hard_smoke']
        specs=game_specs(sampled,args.seed,iteration,config['games_per_secret'])
        for s in specs:s['source']=sources[s['group_index']]
        event('rollout_started',next_iteration=iteration,games=len(specs))
        games,steps=play_complete(policy,engine,sampler,specs,1,config['sampling_temperature'],True,
            progress=lambda **kw:event('rollout_progress',next_iteration=iteration,**kw))
        assert len(games)==config['games_per_iteration'] and all(g.done for g in games)
        for i in range(0,len(specs),config['games_per_secret']):
            assert len({s['secret'] for s in specs[i:i+config['games_per_secret']]})==1
        terminal_rewards=np.array([g.terminal_reward() for g in games])
        descs=[feedback.describe(s.decision) for s in steps]
        eligible=np.array([r['probe_eligible'] for r in descs])
        records=records_from_games(games,steps,specs,feedback)
        penalties=penalty.score_records(records) if penalty else []
        penalty_metrics=summarize_penalties(penalties) if penalty else {}
        rewards=np.array([r['training_reward'] for r in penalties]) if penalty else terminal_rewards.copy()
        assert np.all(rewards<=terminal_rewards) and np.all(rewards>=terminal_rewards-.2-1e-12)
        advantages=rloo_advantages(rewards,config['games_per_secret'])
        terminal_advantages=rloo_advantages(terminal_rewards,config['games_per_secret'])
        for record,scored in zip(records,penalties):
            record['terminal_reward']=record.pop('reward')
            record['reward']=scored['training_reward']
            record['low_information']=scored
        write_json(out/f'rollouts/iteration_{iteration:05d}.json',{'iteration':iteration,
            'sampling_temperature':config['sampling_temperature'],'records':records,
            'rewards':rewards.tolist(),'terminal_win_rewards':terminal_rewards.tolist(),
            'advantages':advantages.tolist(),'terminal_only_advantages':terminal_advantages.tolist(),
            'low_information':penalty_metrics})
        rollout_s=time.monotonic()-iteration_start
        zero=int((advantages.reshape(-1,config['games_per_secret'])==0).all(1).sum())
        terminal_zero=int((terminal_advantages.reshape(-1,config['games_per_secret'])==0).all(1).sum())
        event('rollout',next_iteration=iteration,games=len(games),wins=int(terminal_rewards.sum()),win_rate=float(terminal_rewards.mean()),
            decisions=len(steps),probe_eligible_games=int(eligible.sum()),zero_signal_groups=zero,
            total_groups=len(sampled),zero_signal_fraction=zero/len(sampled),rollout_s=rollout_s,
            terminal_only_zero_signal_fraction=terminal_zero/len(sampled),
            low_information=penalty_metrics,mean_training_reward=float(rewards.mean()),
            **trajectory_metrics(games,steps,feedback))
        update_rng=np.random.default_rng(seed_for('updates',args.seed,iteration));totals=[]
        policy.train()
        for batch_index,indices in enumerate(whole_game_minibatches(steps,len(games),
                config['games_per_optimizer_update'],config['games_per_secret'],update_rng)):
            opt.zero_grad(set_to_none=True);n_aux=int(eligible[indices].sum())
            denominator=6*config['games_per_optimizer_update'];sums={k:0. for k in ['pg','kl','probe','consistency','entropy','clipped']}
            ratio_max=0.;ratio_min=float('inf');max_delta=0.;first_full_delta=0.
            permutation_rng=np.random.default_rng(seed_for('consistency',args.seed,iteration,batch_index))
            lr,head_lr=[g['lr'] for g in opt.param_groups]
            for position,i in enumerate(indices):
                step=steps[i];desc=descs[i]
                lp=policy.log_probs([step.decision],1.)
                lp_sample=(lp/config['sampling_temperature']).log_softmax(-1)
                mask=desc['optimal_mask'] if eligible[i] else np.zeros(64,dtype=bool)
                c=loss_components(lp,lp_sample,step.action,step.log_prob,float(advantages[step.game]),
                    step.ref_log_probs[None,:],mask,config['ppo_ratio_clip'])
                shuffled,inverse=permutation_batch([step.decision],permutation_rng)
                changed=align_log_probs(policy.log_probs(shuffled,1.),inverse)
                consistency=symmetric_kl(lp,changed).sum()
                loss=(c['pg']+config['reference_kl']['coefficient']*c['kl']+
                      config['consistency']['coefficient']*consistency)/denominator
                if n_aux:loss=loss+config['probe_auxiliary']['coefficient']*c['probe']/n_aux
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                if progress['optimizer_step']==0:
                    first_full_delta=max(first_full_delta,float((lp_sample.detach().cpu()[0]-step.sample_log_probs).abs().max()))
                loss.backward()
                for key in ['pg','kl','probe','entropy','clipped']:sums[key]+=float(c[key].detach())
                sums['consistency']+=float(consistency.detach())
                ratio=float(c['ratio'].detach());ratio_max=max(ratio_max,ratio);ratio_min=min(ratio_min,ratio)
                max_delta=max(max_delta,abs(float(lp_sample[0,step.action].detach())-step.log_prob))
                if (position+1)%16==0:
                    event('update_progress',next_iteration=iteration,update_index=batch_index,
                          decisions_done=position+1,decisions_total=len(indices))
            if progress['optimizer_step']==0:
                write_json(out/'preupdate_likelihood_check.json',{'full_distribution_max_abs_delta':first_full_delta,
                    'temperature':config['sampling_temperature'],'rollout_batch':1,'gradient_microbatch':1})
                assert first_full_delta<.02, f'Initial importance-ratio mismatch {first_full_delta}'
            grad=torch.nn.utils.clip_grad_norm_(params,settings['global_gradient_clip_norm'])
            if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradients')
            assert all(p.grad is None for n,p in policy.named_parameters() if '.reference.' in n or n.startswith('reference_head.'))
            opt.step();scheduler.step();progress['optimizer_step']+=1
            row={k:sums[k]/denominator for k in ['pg','kl','consistency']}
            row.update(probe_loss=sums['probe']/n_aux if n_aux else 0.,
                entropy=sums['entropy']/len(indices),clipped_fraction=sums['clipped']/len(indices),
                grad_norm=float(grad),gradient_was_clipped=bool(grad>settings['global_gradient_clip_norm']),
                backbone_learning_rate=lr,head_learning_rate=head_lr,complete_games=config['games_per_optimizer_update'],
                decisions=len(indices),probe_eligible_games=n_aux,ratio_min=ratio_min,ratio_max=ratio_max,
                max_chosen_logp_drift=max_delta)
            row['weighted_probe_loss']=config['probe_auxiliary']['coefficient']*row['probe_loss']
            row['weighted_kl_loss']=config['reference_kl']['coefficient']*row['kl']
            row['weighted_consistency_loss']=config['consistency']['coefficient']*row['consistency']
            row['loss']=row['pg']+row['weighted_kl_loss']+row['weighted_consistency_loss']+row['weighted_probe_loss']
            event('update',next_iteration=iteration,**row);totals.append(row)
        assert digest(policy,True)==progress['reference_sha256'],'Frozen initial reference changed'
        progress['iteration']=iteration;progress['completed_games']+=len(games)
        progress['train_secrets_seen']=sorted(set(progress['train_secrets_seen'])|set(sampled))
        event('rl',games=len(games),cumulative_games=progress['completed_games'],wins=int(terminal_rewards.sum()),
            win_rate=float(terminal_rewards.mean()),zero_signal_fraction=zero/len(sampled),
            terminal_only_zero_signal_fraction=terminal_zero/len(sampled),
            low_information=penalty_metrics,mean_training_reward=float(rewards.mean()),
            unique_train_secrets=len(progress['train_secrets_seen']),probe_eligible_games=int(eligible.sum()),
            rollout_s=rollout_s,update_s=time.monotonic()-iteration_start-rollout_s,
            reference_unchanged=True,total_iterations=config['iterations'],
            **{k:float(np.mean([r[k] for r in totals])) for k in ['loss','pg','kl','consistency','probe_loss',
                'weighted_probe_loss','weighted_kl_loss','weighted_consistency_loss','entropy','clipped_fraction','grad_norm']},
            **trajectory_metrics(games,steps,feedback),memory=memory_metrics())
        # Save before potentially expensive validation so an evaluation failure is resumable.
        progress['pending_validation']=iteration in config['validation']['iterations']
        save('last')
        if progress['pending_validation']:evaluate();save('last')
    event('complete' if progress['iteration']==config['iterations'] else 'intentional_smoke_boundary',
          completed_iterations=progress['iteration'],cumulative_games=progress['completed_games'])


if __name__=='__main__':main()
