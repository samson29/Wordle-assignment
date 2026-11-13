"""Public-only endgame targets and matched diagnostics; never accepts a secret."""
from functools import lru_cache
import numpy as np
import torch
from .environment import Decision, encode


class EndgameFeedback:
    def __init__(self, engine):
        self.engine = engine
        self.state = lru_cache(maxsize=30000)(engine.state)

    def describe(self, decision):
        state = self.state(decision.history)
        possible = {self.engine.answers[a] for a in state.answers}
        ids = [self.engine.gid[w] for w in decision.words]
        n, turns = len(possible), state.remaining_turns
        consistent = np.array([w in possible for w in decision.words])
        singleton = int(np.flatnonzero(consistent)[0]) if n == 1 and consistent.any() else None
        values = None
        if turns == 1:
            values = consistent.astype(np.float64) / n
        elif turns == 2:
            # Under the frozen sampler every next-turn set offers a possible answer.
            values = np.array([len(self.engine.partitions(state, a))/n for a in ids])
        eligible = turns <= 2 or singleton is not None
        target = np.zeros(64, dtype=np.float32)
        if singleton is not None:
            target[singleton] = 1.0
        elif eligible:
            best = np.isclose(values, values.max(), rtol=0, atol=1e-12)
            target[best] = 1.0 / int(best.sum())
        return {"n": n, "turn": len(decision.history)+1, "consistent": consistent,
                "singleton": singleton, "eligible": eligible, "target": target, "values": values}

    def targets(self, decisions):
        descriptions = [self.describe(d) for d in decisions]
        return (np.stack([r["target"] for r in descriptions]),
                np.array([r["eligible"] for r in descriptions]), descriptions)


def components(log_p, actions, old_logp, advantages, ref_logp, targets, clip):
    chosen = log_p.gather(1, actions[:, None]).squeeze(1)
    ratio = (chosen-old_logp).exp()
    pg = -torch.minimum(ratio*advantages, ratio.clamp(1-clip,1+clip)*advantages)
    # Generalized KL equals categorical KL for normalized distributions.
    # The -p+q terms remove the floating-point residual gradient at p==q;
    # double precision avoids cancellation when the policies are close.
    lp, lq = log_p.double(), ref_logp.double()
    kl = (lp.exp()*(lp-lq-1)+lq.exp()).sum(-1).to(log_p.dtype)
    ce = -(targets*log_p).sum(-1)
    entropy = -(log_p.exp()*log_p).sum(-1)
    return {"pg":pg,"kl":kl,"solver_ce":ce,"entropy":entropy,"ratio":ratio,
            "clipped":(ratio.detach()-1).abs()>clip}


def whole_game_minibatches(steps, n_games, games_per_batch, group, rng):
    if n_games % group or games_per_batch % group or n_games % games_per_batch:
        raise ValueError("Whole groups and complete games must fit each optimizer minibatch")
    group_order = rng.permutation(n_games//group)
    ordered_games = [int(g*group+j) for g in group_order for j in range(group)]
    for start in range(0,n_games,games_per_batch):
        games = set(ordered_games[start:start+games_per_batch])
        indices = [i for i,s in enumerate(steps) if s.game in games]
        assert len({steps[i].game for i in indices})==games_per_batch
        yield indices


def trajectory_metrics(games, steps, feedback):
    all_rows = [feedback.describe(s.decision) for s in steps]
    closing = {str(t):{"opportunities":0,"correct":0} for t in range(1,7)}
    hard=[]; turn5={}; after_probe=[]
    for step,row in zip(steps,all_rows):
        if row['singleton'] is not None:
            c=closing[str(row['turn'])];c['opportunities']+=1;c['correct']+=int(step.action==row['singleton'])
        if row['turn']==5:
            turn5[step.game]=(step,row)
            if row['n']>2:
                q=row['values'];best=float(q.max());consistent=row['consistent']
                best_consistent=float(q[consistent].max()) if consistent.any() else 0.0
                hard.append({'chosen_value':float(q[step.action]),'best_value':best,
                    'regret':max(0.0,best-float(q[step.action])),
                    'optimal':bool(q[step.action]>=best-1e-12),
                    'probe_required':best>best_consistent+1e-12,
                    'guarantee_available':best>=1-1e-12,
                    'guarantee_missed':bool(best>=1-1e-12 and q[step.action]<1-1e-12),
                    'chose_probe':not bool(consistent[step.action])})
        if row['turn']==6 and row['singleton'] is not None and step.game in turn5:
            previous,desc=turn5[step.game]
            if desc['n']>1 and not desc['consistent'][previous.action]:
                after_probe.append(int(step.action==row['singleton']))
    for c in closing.values():c['accuracy']=c['correct']/c['opportunities'] if c['opportunities'] else None
    opportunities=sum(c['opportunities'] for c in closing.values());correct=sum(c['correct'] for c in closing.values())
    result={'closing':{'opportunities':opportunities,'correct':correct,
        'accuracy':correct/opportunities if opportunities else None,'by_turn':closing},
        'turn5':{'hard_decisions':len(hard)},
        'closing_after_turn5_probe':{'opportunities':len(after_probe),'correct':sum(after_probe),
            'accuracy':float(np.mean(after_probe)) if after_probe else None}}
    if hard:
        for key in hard[0]:result['turn5'][key+'_mean']=float(np.mean([x[key] for x in hard]))
        for key in ['probe_required','guarantee_available','guarantee_missed']:
            result['turn5'][key+'_count']=int(sum(x[key] for x in hard))
        needed=[x for x in hard if x['probe_required']]
        result['turn5']['optimal_when_probe_required']=float(np.mean([x['optimal'] for x in needed])) if needed else None
    return result


def fixed_panel_metrics(policy, panel, feedback, batch_size=8):
    decisions=[Decision.from_dict(r) for r in panel]
    logp=policy.score(decisions,batch_size);p=logp.exp().numpy()
    q,eligible,descriptions=feedback.targets(decisions)
    assert eligible.all()
    chosen=p.argmax(1);optimal=[];regret=[];mass=[];expected=[]
    for i,row in enumerate(descriptions):
        values=row['values'];best=values.max()
        optimal.append(values[chosen[i]]>=best-1e-12)
        regret.append(float(best-values[chosen[i]]))
        mass.append(float(p[i,q[i]>0].sum()))
        expected.append(float(np.dot(p[i],values)))
    return {'states':len(panel),'optimal_action_accuracy':float(np.mean(optimal)),
        'mean_solver_regret':float(np.mean(regret)),'optimal_action_probability':float(np.mean(mass)),
        'mean_expected_solver_value':float(np.mean(expected)),
        'solver_cross_entropy':float(-(torch.tensor(q)*logp).sum(-1).mean()),
        'decisions':[{'chosen_word':d.words[int(chosen[i])],'regret':regret[i],
            'optimal_probability':mass[i]} for i,d in enumerate(decisions)]}
