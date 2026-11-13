"""Complete validation trajectories, all-turn order audits and public diagnostics."""
from collections import defaultdict
from pathlib import Path
import json
import time
import numpy as np
import torch
from .environment import Decision, seed_for
from .probe_study import play_complete, records_from_games, write_json
from .endgame_feedback import trajectory_metrics, fixed_panel_metrics
from .planner import ChallengePlanner
from .readout_diagnostics import load_panel
from .low_information import LowInformationPenalty, summarize_penalties


def ci(values):
    a = np.asarray(values); a = a[np.isfinite(a)]
    return np.quantile(a, [.025, .975]).tolist() if len(a) else None


def cluster_ratio(num, den, answer_ids, weights):
    stats = np.zeros((weights.shape[1], 2))
    np.add.at(stats, answer_ids, np.column_stack([num, den]))
    total = stats.sum(0); draws = weights @ stats; valid = draws[:, 1] > 0
    return {'numerator': float(total[0]), 'denominator': int(total[1]),
            'mean': float(total[0]/total[1]) if total[1] else None,
            'answer_bootstrap_95pct': ci(draws[valid, 0]/draws[valid, 1])}


def calibration(rows, weights):
    stats = np.zeros((weights.shape[1], 15, 3))
    for r in rows:
        b = min(14, int(r['confidence']*15))
        stats[r['answer_index'], b] += [1, r['confidence'], r['closing_aware_best']]
    total=stats.sum(0); drawn=(weights@stats.reshape(weights.shape[1],-1)).reshape(len(weights),15,3)
    den=drawn[:,:,0].sum(1); valid=den>0
    return {'decisions':len(rows),'ECE':float(abs(total[:,2]-total[:,1]).sum()/len(rows)) if rows else None,
        'answer_bootstrap_95pct':ci(abs(drawn[valid,:,2]-drawn[valid,:,1]).sum(1)/den[valid]),
        'reliability_bins':[{'low':i/15,'high':(i+1)/15,'count':int(n),
            'mean_confidence':float(c/n) if n else None,'accuracy':float(a/n) if n else None}
            for i,(n,c,a) in enumerate(total)]}


def attach_penalty_metrics(report,records,engine,out,coefficient=.1):
    scorer=LowInformationPenalty(engine,coefficient)
    scores=scorer.score_records(records)
    report['low_information']=summarize_penalties(scores)
    n=max(r['answer_index'] for r in records)+1
    draws=np.random.default_rng(722).integers(0,n,size=(2000,n))
    weights=np.array([np.bincount(d,minlength=n) for d in draws],dtype=float)
    rows=[{'answer_index':record['answer_index'],**turn}
          for record,score in zip(records,scores) for turn in score['turns']]
    def rates(rs):
        return cluster_ratio([r['penalized'] for r in rs],[r['nonwinning_unresolved'] for r in rs],
            [r['answer_index'] for r in rs],weights)
    report['low_information']['penalized_rate_with_ci']=rates(rows)
    report['low_information']['penalized_rate_by_turn_with_ci']={str(t):rates([r for r in rows if r['turn']==t]) for t in (4,5)}
    write_json(Path(out)/'low_information_games.json',[
        {'seed':r['seed'],'answer_index':r['answer_index'],**s} for r,s in zip(records,scores)])
    return scores


def evaluate_complete(policy, engine, sampler, secrets, seeds, feedback, panel, out,
                      event, diagnostic_data=None, batch_size=8, quick=False,penalty_coefficient=.1):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    all_games=[];all_records=[];all_rows=[];order_rows=[];global_steps=[]
    planner=ChallengePlanner(engine,sampler);solver_cache={}
    draws=np.random.default_rng(722).integers(0,len(secrets),size=(2000,len(secrets)))
    weights=np.array([np.bincount(row,minlength=len(secrets)) for row in draws],dtype=float)
    for seed in seeds:
        rng=np.random.default_rng(seed)
        specs=[{'secret':str(secret),'answer_index':i,'seed':seed,
                'environment_seed':int(rng.integers(2**63)),
                'action_seed':seed_for('unused-greedy',seed,i)} for i,secret in enumerate(secrets)]
        games,steps=play_complete(policy,engine,sampler,specs,batch_size,1.,greedy=True,
            progress=lambda **kw:event('validation_gameplay',seed=seed,**kw))
        records=records_from_games(games,steps,specs,feedback)
        write_json(out/f'games_seed{seed}.json',records)
        offset=len(all_games)
        from dataclasses import replace
        global_steps.extend(replace(s,game=s.game+offset) for s in steps)
        all_games.extend(games);all_records.extend(records)
        decisions=[s.decision for s in steps]
        original=np.stack([s.raw_log_probs.numpy() for s in steps])
        original_p=np.exp(original.astype(np.float64))
        turns=np.array([len(d.history)+1 for d in decisions])
        answers=np.array([s.game for s in steps])
        descriptions=[feedback.describe(d) for d in decisions]
        ns=np.array([r['n'] for r in descriptions])
        # Preserve the exact seed/turn/batch boundaries used during gameplay.
        batches=[]
        for turn in range(1,7):
            ids=np.flatnonzero(turns==turn)
            for start in range(0,len(ids),batch_size):batches.append(ids[start:start+batch_size])
        n=len(steps);permutation_rng=np.random.default_rng(20261117)
        permutations={'repeat':np.tile(np.arange(64),(n,1)),
            'reverse':np.tile(np.arange(63,-1,-1),(n,1)),
            'random1':np.array([permutation_rng.permutation(64) for _ in range(n)]),
            'random2':np.array([permutation_rng.permutation(64) for _ in range(n)])}
        phase_rows={}
        for phase,orders in permutations.items():
            changed=[Decision(d.history,tuple(d.words[int(i)] for i in order))
                     for d,order in zip(decisions,orders)]
            lp=np.empty_like(original);last_log=time.monotonic()
            for ids in batches:
                lp[ids]=policy.score([changed[int(i)] for i in ids],batch_size).numpy()
                if time.monotonic()-last_log>20:
                    event('validation_order_progress',seed=seed,phase=phase,done=int(ids[-1])+1,total=n)
                    last_log=time.monotonic()
            aligned=np.take_along_axis(np.exp(lp.astype(np.float64)),np.argsort(orders,axis=1),axis=1)
            chosen=orders[np.arange(n),lp.argmax(1)]
            flips=chosen!=original.argmax(1);tv=np.abs(original_p-aligned).sum(1)/2
            (out/'order').mkdir(exist_ok=True)
            np.savez_compressed(out/'order'/f'{seed}_{phase}.npz',log_probabilities=lp,
                original_log_probabilities=original,permutations=orders,flips=flips,tv=tv,
                answer_index=answers,turn=turns,remaining_answers=ns)
            phase_rows[phase]=(flips,tv)
            event('validation_order_complete',seed=seed,phase=phase,decisions=n,
                  flip_rate=float(flips.mean()),mean_tv=float(tv.mean()))
        for i in range(n):
            order_rows.append({'seed':seed,'answer_index':int(answers[i]),'turn':int(turns[i]),'n':int(ns[i]),
                'flips':sum(int(phase_rows[k][0][i]) for k in ['reverse','random1','random2']),
                'tv':sum(float(phase_rows[k][1][i]) for k in ['reverse','random1','random2']),
                'repeat_flip':int(phase_rows['repeat'][0][i]),'repeat_tv':float(phase_rows['repeat'][1][i])})
        last_log=time.monotonic()
        for i,(step,desc) in enumerate(zip(steps,descriptions)):
            d=step.decision;state=feedback.state(d.history);chosen=engine.gid[d.words[step.action]]
            # Public solver only; no secret is passed to the label calculation.
            if d.history not in solver_cache:
                best,values=planner.choose(state,tuple(sampler.sample(state)))
                solver_cache[d.history]=(best,{r['guess_id']:r['continuation_win_probability'] for r in values})
            best,values=solver_cache[d.history]
            if chosen not in values:values[chosen]=planner._action_value(state,chosen,None)
            q,bq=float(values[chosen]),float(values[best]);assert q<=bq+1e-10
            singleton=desc['singleton'];is_close=singleton is not None
            optimal=q>=bq-1e-12
            hard=desc['turn']==5 and desc['n']>2
            eligible=desc['probe_eligible']
            mass=float(original_p[i,desc['optimal_mask']].sum()) if desc['turn']==5 else None
            row={'seed':seed,'answer_index':step.game,'turn':desc['turn'],'n':desc['n'],
                'confidence':float(original_p[i,step.action]),
                'closing_aware_best':int(step.action==singleton) if is_close else int(optimal),
                'closing_opportunity':int(is_close),'closing_correct':int(is_close and step.action==singleton),
                'hard_turn5':int(hard),'probe_required':int(eligible),
                'probe_optimal':int(eligible and optimal),'probe_regret':max(0,bq-q) if eligible else 0,
                'probe_probability':mass if eligible else 0.,
                'probe_nll':float(-torch.logsumexp(step.raw_log_probs[torch.as_tensor(desc['optimal_mask'])],0)) if eligible else 0.,
                'guarantee_available':int(hard and bq>=1-1e-12),
                'guarantee_missed':int(hard and bq>=1-1e-12 and q<1-1e-12),
                'guess_discarded_guarantee':int(hard and bq>=1-1e-12 and q<1-1e-12 and desc['consistent'][step.action]),
                'guarantee_preserving_probe':int(hard and q>=1-1e-12 and not desc['consistent'][step.action]),
                'won_after_guarantee_probe':int(hard and q>=1-1e-12 and not desc['consistent'][step.action] and games[step.game].won),
                'chosen_q':q,'best_q':bq,'regret':max(0,bq-q),
                'chosen_word':d.words[step.action],'chosen_consistent':bool(desc['consistent'][step.action]),
                'game_won':bool(games[step.game].won)}
            all_rows.append(row)
            if time.monotonic()-last_log>20:
                event('validation_public_labels',seed=seed,done=i+1,total=n);last_log=time.monotonic()
        event('validation_seed_complete',seed=seed,games=len(games),wins=sum(g.won for g in games))
    answer_ids=np.array([r['answer_index'] for r in all_records]);ones=np.ones(len(all_records))
    wins=np.array([r['won'] for r in all_records]);lengths=np.array([len(r['history']) for r in all_records])
    wr=cluster_ratio(wins,ones,answer_ids,weights)
    result={'games':len(all_games),'unique_answers':len(secrets),'wins':int(wins.sum()),
        'win_rate':wr['mean'],'answer_bootstrap_95pct':wr['answer_bootstrap_95pct'],
        'mean_guesses_wins':cluster_ratio(lengths*wins,wins,answer_ids,weights),
        'failed_requests':0,'decisions':len(all_rows),
        'by_seed':[{'seed':seed,'games':len(secrets),'wins':sum(r['won'] for r in all_records if r['seed']==seed)} for seed in seeds],
        'guess_distribution':{str(t):cluster_ratio(wins&(lengths==t),ones,answer_ids,weights) for t in range(1,7)},
        **trajectory_metrics(all_games,global_steps,feedback)}
    result['guess_distribution']['failed']=cluster_ratio(~wins,ones,answer_ids,weights)
    ids=np.array([r['answer_index'] for r in all_rows])
    def ratio(rows,num,den):
        return cluster_ratio([r[num] for r in rows],[r[den] for r in rows],
                             [r['answer_index'] for r in rows],weights)
    result['probe_required']={name:ratio(all_rows,num,'probe_required') for name,num in
        [('optimal_accuracy','probe_optimal'),('mean_regret','probe_regret'),
         ('optimal_probability','probe_probability'),('validation_probe_loss','probe_nll')]}
    result['missed_turn5_guarantees']=ratio(all_rows,'guarantee_missed','guarantee_available')
    result['turn5']['guarantees_discarded_by_guess']=sum(r['guess_discarded_guarantee'] for r in all_rows)
    result['turn5']['guarantees_discarded_by_bad_probe']=sum(r['guarantee_missed']-r['guess_discarded_guarantee'] for r in all_rows)
    result['win_after_guarantee_preserving_probe']=ratio(all_rows,'won_after_guarantee_probe','guarantee_preserving_probe')
    result['public_value_by_turn']={str(t):{'decisions':len(rs),
        'mean_regret':float(np.mean([r['regret'] for r in rs])) if rs else None,
        'suboptimal_decisions':sum(r['regret']>1e-12 for r in rs)}
        for t in range(1,7) for rs in [[r for r in all_rows if r['turn']==t]]}
    result['closing_with_ci']=ratio(all_rows,'closing_correct','closing_opportunity')
    result['calibration']=calibration(all_rows,weights)
    result['calibration']['definition']='Best public fixed-continuation solver value, with singleton immediate-closing rule. Turn5/6 exact for public uniform posterior; earlier values not global optimality.'
    result['calibration_by_turn']={str(t):calibration([r for r in all_rows if r['turn']==t],weights) for t in range(1,7)}
    bins=[('1',1,1),('2',2,2),('3-5',3,5),('6-20',6,20),('21+',21,2315)]
    result['calibration_by_remaining_answers']={label:calibration([r for r in all_rows if lo<=r['n']<=hi],weights) for label,lo,hi in bins}
    def order_summary(rows):
        ids=[r['answer_index'] for r in rows];n=len(rows)
        return {'decisions':n,'comparisons':3*n,'permutations_per_decision':3,
            'flip_rate':cluster_ratio([r['flips'] for r in rows],[3]*n,ids,weights),
            'mean_tv':cluster_ratio([r['tv'] for r in rows],[3]*n,ids,weights),
            'same_order_flip_rate':cluster_ratio([r['repeat_flip'] for r in rows],[1]*n,ids,weights),
            'same_order_mean_tv':cluster_ratio([r['repeat_tv'] for r in rows],[1]*n,ids,weights),
            'any_flip_rate':cluster_ratio([r['flips']>0 for r in rows],[1]*n,ids,weights)}
    result['order']=order_summary(order_rows)
    result['order']['by_turn']={str(t):order_summary([r for r in order_rows if r['turn']==t]) for t in range(1,7)}
    result['order']['by_remaining_answers']={label:order_summary([r for r in order_rows if lo<=r['n']<=hi]) for label,lo,hi in bins}
    result['fixed_turn5_panel']=fixed_panel_metrics(policy,panel,feedback,batch_size)
    if diagnostic_data:
        ds=load_panel(diagnostic_data);lp=policy.score([Decision.from_dict(s) for s in ds],batch_size)
        losses=[float(-torch.logsumexp(lp[i,s['accepted']],0)) for i,s in enumerate(ds)]
        result['validation_sl_diagnostic']={'states':len(ds),'accepted_set_loss':float(np.mean(losses)),
            'population':'Fixed teacher-validation panel; not full-game RL loss',
            'by_turn':{str(t):float(np.mean([losses[i] for i,s in enumerate(ds) if len(s['history'])==t-1]))
                       for t in range(1,7) if any(len(s['history'])==t-1 for s in ds)}}
    result['evaluation_s']=time.monotonic()-started
    if penalty_coefficient is not None:
        attach_penalty_metrics(result,all_records,engine,out,penalty_coefficient)
    write_json(out/'decision_metrics.json',all_rows);write_json(out/'order_metrics.json',order_rows)
    write_json(out/'metrics.json',result)
    return result
