"""Verify complete train-only mining and freeze a secret-level hard pool."""
import argparse
from pathlib import Path
import json
import hashlib
from collections import Counter
from _common import environment
from wordle_k64.environment import Decision, feedback as actual_feedback
from wordle_k64.probe_study import ProbeFeedback, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--shards',type=int,default=4);a=p.parse_args()
    if a.shards < 1:p.error('--shards must be positive')
    split,engine,sampler=environment();feedback=ProbeFeedback(engine)
    complete=[json.loads((a.out/f'shard_{i:02d}/complete.json').read_text()) for i in range(a.shards)]
    assert len({r['policy_sha256'] for r in complete})==1
    assert all(r['training_updates']==0 and r['split']=='train' for r in complete)
    assert all(r['seeds']==complete[0]['seeds'] for r in complete)
    seen=set();hard=Counter();histories=set();total=0;hardgames=0;six=0;hashes={}
    for path in sorted(a.out.glob('shard_*/games_*.json')):
        hashes[str(path.relative_to(a.out))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for game in json.loads(path.read_text()):
            secret=game['secret'];assert secret in split['train']
            key=(secret,game['mining_seed']);assert key not in seen;seen.add(key)
            assert 1<=len(game['steps'])<=6
            assert game['won'] or len(game['steps'])==6
            history=[];eligible=False
            for step in game['steps']:
                assert step['history']==history
                d=Decision.from_dict(step)
                assert set(d.words)=={engine.guesses[g] for g in sampler.sample(engine.state(d.history))}
                chosen=d.words[step['action']];assert chosen==step['chosen_word']
                history.append([chosen,actual_feedback(chosen,secret)])
                desc=feedback.describe(d);assert desc['probe_eligible']==step['probe_eligible']
                if desc['probe_eligible']:eligible=True;histories.add(d.history)
            assert history==game['history'];assert game['won']==(history[-1][1]=='GGGGG')
            total+=1;six+=len(history)==6
            if eligible:hard[secret]+=1;hardgames+=1
    seeds=complete[0]['seeds']
    assert seen=={(secret,seed) for secret in split['train'] for seed in seeds}
    assert total==1565*len(seeds)
    assert len(hard)>=16, 'Insufficient hard pool: collect the planned additional train-only seeds'
    pool={'status':'frozen','training_split_only':True,'secrets':sorted(hard),
        'unique_hard_secrets':len(hard),'complete_games':total,'six_turn_games':six,
        'probe_required_games':hardgames,'unique_probe_required_histories':len(histories),
        'occurrences_per_secret':dict(sorted(hard.items())),
        'mining_seeds':seeds,'policy_sha256':complete[0]['policy_sha256'],
        'rule':'turn5, N>2, best public Q5 strictly greater than the offered consistent guess Q5',
        'selection_weight':'uniform per secret, not per occurrence',
        'source_hashes':hashes,'gpu_hours':sum(r['gpu_hours'] for r in complete)}
    write_json(a.out/'hard_train_secrets.json',pool)
    print(json.dumps({k:v for k,v in pool.items() if k not in ['secrets','occurrences_per_secret','source_hashes']},indent=2))


if __name__=='__main__':main()
