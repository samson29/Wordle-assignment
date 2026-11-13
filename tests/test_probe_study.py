import json
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict
import numpy as np
import torch
from wordle_k64.environment import Decision, feedback as scalar_feedback
from wordle_k64.contracts import load_contract
from wordle_k64.probe_study import (ProbeFeedback, rloo_advantages, loss_components,
    secret_schedule, play_complete, game_specs, records_from_games)
from wordle_k64.probe_validation import cluster_ratio


def panel():
    return json.loads((Path(__file__).resolve().parents[1]/'data/rl_turn5_panel.json').read_text())['states']


def test_q5_matches_explicit_public_two_turn_game_enumeration(env):
    engine,sampler=env;public=ProbeFeedback(engine)
    for row in panel()[:5]:
        d=Decision.from_dict(row);desc=public.describe(d)
        possible=[engine.answers[i] for i in engine.state(d.history).answers]
        for word,q in zip(d.words,desc['values']):
            # Independent scalar BGY implementation and actual final candidate policy.
            buckets=defaultdict(list)
            for hypothetical in possible:buckets[scalar_feedback(word,hypothetical)].append(hypothetical)
            wins=0
            for pattern,answers in buckets.items():
                if pattern=='GGGGG':wins+=len(answers);continue
                child=engine.state(d.history+((word,pattern),))
                offered=[engine.guesses[g] for g in sampler.sample(child)]
                consistent=[w for w in offered if w in answers]
                assert len(consistent)==1
                wins+=sum(a==consistent[0] for a in answers)
            assert abs(q-wins/len(possible))<1e-12


def test_probe_gradient_survives_all_loss_group_and_ties(env):
    engine,_=env;f=ProbeFeedback(engine)
    d=Decision.from_dict(next(r for r in panel() if r['probe_required']))
    desc=f.describe(d);bad=int(np.flatnonzero(desc['consistent'])[0])
    z=torch.zeros(1,64);z[0,bad]=7;z.requires_grad_()
    lp=z.log_softmax(-1);sample=(lp/1.2).log_softmax(-1)
    a=rloo_advantages([0]*8,8);assert not a.any()
    c=loss_components(lp,sample,bad,float(sample[0,bad].detach()),a[0],lp.detach(),desc['optimal_mask'])
    assert c['pg']==0 and abs(float(c['kl']))<1e-10 and abs(float(c['ratio'])-1)<1e-7
    loss=.1*c['probe'];loss.backward();assert z.grad[0,bad]>0
    before=lp.exp().detach();after=(z.detach()-z.grad).softmax(-1)
    assert after[0,desc['optimal_mask']].sum()>before[0,desc['optimal_mask']].sum()
    assert after[0,bad]<before[0,bad]
    # Target is a set: dividing mass among tied optimal words does not change loss.
    mask=np.array([True,True,False]);p1=torch.tensor([[.2,.6,.2]]).log();p2=torch.tensor([[.4,.4,.2]]).log()
    assert torch.allclose(-torch.logsumexp(p1[0,mask],0),-torch.logsumexp(p2[0,mask],0))


def test_reference_temperature_is_untempered_and_ratios_are_tempered():
    raw=torch.tensor([[0.,2.,-1.]],requires_grad=True).log_softmax(-1)
    sample=(raw/1.2).log_softmax(-1)
    c=loss_components(raw,sample,1,float(sample[0,1].detach()),1.,raw.detach(),[False]*3)
    assert abs(float(c['ratio'])-1)<1e-7 and abs(float(c['kl']))<1e-10
    assert abs(float(raw[0,1]-sample[0,1]))>.01
    c['pg'].backward()


def test_rloo_and_matched_secret_schedules():
    split,_=load_contract();hard=split['train'][:30]
    for iteration in range(1,25):
        c,_=secret_schedule(split['train'],hard,0,19,iteration)
        d,sources=secret_schedule(split['train'],hard,.5,19,iteration)
        assert c[:8]==d[:8] and len(set(d))==16 and set(d[8:])<=set(hard)
        assert sources.count('hard')==8
        assert not set(d)&set(split['validation']+split['test'])
    a=rloo_advantages([1,0,0,0,0,0,0,0],8)
    assert a[0]==1 and np.allclose(a[1:],-1/7) and abs(a.sum())<1e-10


class ToyPolicy:
    @contextmanager
    def reference(self):yield
    def score(self,decisions,batch_size=1,temperature=1.):
        assert all(set(d.to_dict())=={'history','words'} for d in decisions)
        logits=torch.tensor([[sum(ord(c) for c in w)%13/4 for w in d.words] for d in decisions])
        return (logits/temperature).log_softmax(-1)


def test_complete_games_per_game_rng_and_reference_recording(env):
    engine,sampler=env;split,_=load_contract();policy=ToyPolicy();fb=ProbeFeedback(engine)
    specs=game_specs(split['train'][:2],77,1,2)
    games,steps=play_complete(policy,engine,sampler,specs,1,1.2,True)
    shorter,shortsteps=play_complete(policy,engine,sampler,specs[:2],1,1.2,True)
    assert [g.history for g in games[:2]]==[g.history for g in shorter]
    assert all(g.done and (g.won or len(g.history)==6) for g in games)
    assert len(steps)==sum(len(g.history) for g in games)
    for step in steps:
        assert torch.equal(step.raw_log_probs,step.ref_log_probs)
        assert abs(float(step.sample_log_probs[step.action])-step.log_prob)<1e-8
    records=records_from_games(games,steps,specs,fb)
    assert len(records)==4 and all(len(r['steps'])==len(r['history']) for r in records)


def test_bootstrap_empty_stratum_and_cluster_denominators():
    weights=np.array([[1,1],[2,0],[0,2]],dtype=float)
    empty=cluster_ratio([],[],[],weights)
    assert empty['mean'] is None and empty['answer_bootstrap_95pct'] is None
    r=cluster_ratio([1,0,1],[1,1,1],[0,0,1],weights)
    assert r['numerator']==2 and r['denominator']==3 and r['mean']==2/3
