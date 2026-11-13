import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from wordle_k64.environment import Decision, decode
from wordle_k64.low_information import LowInformationPenalty, summarize_penalties
from wordle_k64.probe_study import rloo_advantages, loss_components

ROOT=Path(__file__).resolve().parents[1]


def hard_decision(engine):
    rows=json.loads((ROOT/'data/rl_turn5_panel.json').read_text())['states']
    for row in rows:
        d=Decision.from_dict(row);s=engine.state(d.history);n=len(s.answers)
        for i,w in enumerate(d.words):
            buckets=engine.partitions(s,engine.gid[w])
            for pattern,answers in buckets.items():
                if len(answers)==n-1 and n>2 and LowInformationPenalty(engine).describe(d,i,decode(pattern))['penalized']:
                    return d,i,decode(pattern)
    raise AssertionError('No one-elimination example')


def test_actual_one_elimination_is_penalized_but_zero_cannot_escape(env):
    engine,_=env;penalty=LowInformationPenalty(engine);d,a,pattern=hard_decision(engine)
    row=penalty.describe(d,a,pattern)
    assert row['turn']==5 and row['eliminated']==1 and row['penalty']==.1
    assert row['answers_after']>1 and row['better_option_available']
    state=engine.state(d.history)
    for i,w in enumerate(d.words):
        buckets=engine.partitions(state,engine.gid[w])
        if len(buckets)==1:
            zero=penalty.describe(d,i,decode(next(iter(buckets))))
            assert zero['eliminated']==0 and zero['penalty']==.1
            break
    else:raise AssertionError('Expected an uninformative offered distractor')


def test_winning_and_singleton_producing_guesses_are_exempt(env):
    engine,_=env;penalty=LowInformationPenalty(engine);d,a,pattern=hard_decision(engine)
    state=engine.state(d.history)
    actual_possible={engine.answers[i] for i in state.answers}
    guess=next(i for i,w in enumerate(d.words) if w in actual_possible)
    assert penalty.describe(d,guess,'GGGGG')['reason']=='winning_guess'
    found=False
    for path in [ROOT/'tests/reward_games.json']:
        for game in json.loads(path.read_text()):
            for i,step in enumerate(game['steps']):
                if i+1 not in (4,5):continue
                desc=penalty.describe(Decision.from_dict(step),step['action'],game['history'][i][1])
                if desc['answers_before']==2 and desc['answers_after']==1 and desc['observed_feedback']!='GGGGG':
                    assert desc['eliminated']==1 and not desc['penalized'];found=True
    assert found


def test_best_available_information_choice_is_not_punished(env):
    engine,_=env;penalty=LowInformationPenalty(engine);d,a,pattern=hard_decision(engine)
    state=engine.state(d.history);n=len(state.answers);word=d.words[a]
    expected=lambda w:sum(len(b)**2 for b in engine.partitions(state,engine.gid[w]).values())/n
    threshold=expected(word)
    others=[w for w in engine.guesses if w!=word and expected(w)>=threshold-1e-12][:63]
    # A hypothetical provided set, solely to exercise the unavailable-better-option boundary.
    limited=Decision(d.history,tuple([word]+others))
    row=penalty.describe(limited,0,pattern)
    assert row['eliminated']==1 and not row['penalized']
    assert row['reason']=='no_strictly_better_expected_reduction_offered'


def test_complete_game_only_secret_independence_and_reward_bounds(env):
    engine,_=env;penalty=LowInformationPenalty(engine)
    path=ROOT/'tests/reward_games.json'
    records=json.loads(path.read_text())
    first=penalty.score_records(records)
    changed=copy.deepcopy(records)
    for r in changed:r['secret']='IGNORED';r['won']=not r['won'];r['reward']=12345
    assert penalty.score_records(changed)==first
    assert any(r['penalty_count'] for r in first)
    for r in first:
        assert r['penalty_count'] in [0,1,2]
        assert r['terminal_win_reward']-.2-1e-12<=r['training_reward']<=r['terminal_win_reward']
    unfinished=records[0]['history'][:3]
    with pytest.raises(ValueError):penalty.score_complete(unfinished,[])
    summary=summarize_penalties(first)
    assert summary['penalized_count']==sum(r['penalty_count'] for r in first)


def test_penalty_creates_relative_signal_and_decreases_penalized_action_probability():
    assert np.array_equal(rloo_advantages([.9]*8,8),np.zeros(8))
    rewards=[1.,.9,1.,1.,1.,1.,1.,1.]
    a=rloo_advantages(rewards,8)
    assert a[1]<0 and a[0]>0 and abs(a.sum())<1e-12
    z=torch.zeros(1,64,requires_grad=True);lp=z.log_softmax(-1);sample=(lp/1.2).log_softmax(-1)
    c=loss_components(lp,sample,7,float(sample[0,7].detach()),float(a[1]),lp.detach(),[False]*64)
    c['pg'].backward()
    assert z.grad[0,7]>0
    assert (z.detach()-z.grad).softmax(-1)[0,7]<lp.detach().exp()[0,7]
