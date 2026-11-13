"""Postgame trajectory penalty using only observed public histories and options."""
from functools import lru_cache
import numpy as np
from .environment import Decision


class LowInformationPenalty:
    def __init__(self, engine, coefficient=.1):
        if not 0 < coefficient < .5:
            raise ValueError('Two late-turn penalties must remain smaller than a win reward')
        self.engine=engine;self.coefficient=coefficient
        self.state=lru_cache(maxsize=30000)(engine.state)
        self.expected_cache={}

    def describe(self, decision, action, observed_pattern):
        """Public transition only. Caller scores these after the full game finishes."""
        before=self.state(decision.history)
        chosen=decision.words[action]
        after=self.state(decision.history+((chosen,observed_pattern),))
        n0,n1=len(before.answers),len(after.answers)
        assert 1<=n1<=n0
        key=(decision.history,tuple(sorted(decision.words)))
        if key not in self.expected_cache:
            self.expected_cache[key]={w:sum(len(b)**2 for b in self.engine.partitions(
                before,self.engine.gid[w]).values())/n0 for w in decision.words}
        expected=self.expected_cache[key]
        best=min(expected.values());chosen_expected=expected[chosen]
        better=chosen_expected>best+1e-12
        turn=len(decision.history)+1
        unresolved=observed_pattern!='GGGGG' and n1>1
        raw_low=turn in (4,5) and unresolved and n0-n1<=1
        penalized=raw_low and better
        if turn not in (4,5):reason='outside_target_turns'
        elif observed_pattern=='GGGGG':reason='winning_guess'
        elif n0==1:reason='already_determined'
        elif n1==1:reason='answer_determined_afterward'
        elif n0-n1>1:reason='eliminated_more_than_one'
        elif not better:reason='no_strictly_better_expected_reduction_offered'
        else:reason='avoidable_zero_or_one_elimination'
        return {'turn':turn,'chosen_word':chosen,'observed_feedback':observed_pattern,
            'answers_before':n0,'answers_after':n1,'eliminated':n0-n1,
            'nonwinning_unresolved':bool(unresolved),'raw_low_information':bool(raw_low),
            'chosen_expected_answers_after':chosen_expected,'best_expected_answers_after':best,
            'better_option_available':bool(better),'penalized':bool(penalized),
            'penalty':self.coefficient if penalized else 0.,'reason':reason,
            'best_information_words':sorted(w for w,v in expected.items() if v<=best+1e-12)}

    def score_complete(self, history, decisions_and_actions):
        """No actual-secret argument: final public history proves game completion."""
        history=tuple(tuple(x) for x in history)
        if not 1<=len(history)<=6 or (len(history)<6 and history[-1][1]!='GGGGG'):
            raise ValueError('Penalty must be computed after a complete game')
        if len(decisions_and_actions)!=len(history) or any(p=='GGGGG' for _,p in history[:-1]):
            raise ValueError('Malformed complete trajectory')
        rows=[]
        for i,((d,action),(word,pattern)) in enumerate(zip(decisions_and_actions,history)):
            assert d.history==history[:i] and d.words[action]==word
            if i+1 in (4,5):rows.append(self.describe(d,action,pattern))
        penalty=sum(r['penalty'] for r in rows)
        assert 0<=penalty<=2*self.coefficient+1e-12
        win=float(history[-1][1]=='GGGGG')
        return {'terminal_win_reward':win,'low_information_penalty':penalty,
                'training_reward':win-penalty,'penalty_count':sum(r['penalized'] for r in rows),
                'turns':rows}

    def score_records(self, records):
        return [self.score_complete(r['history'],
            [(Decision.from_dict(s),s['action']) for s in r['steps']]) for r in records]


def summarize_penalties(scores):
    rows=[r for g in scores for r in g['turns']]
    def summarize(rs):
        eligible=[r for r in rs if r['nonwinning_unresolved']]
        penalized=sum(r['penalized'] for r in rs)
        return {'decisions':len(rs),'nonwinning_unresolved_decisions':len(eligible),
            'raw_zero_or_one_elimination_count':sum(r['raw_low_information'] for r in rs),
            'penalized_count':penalized,
            'penalized_rate':penalized/len(eligible) if eligible else None,
            'one_elimination_penalties':sum(r['penalized'] and r['eliminated']==1 for r in rs),
            'zero_elimination_penalties':sum(r['penalized'] and r['eliminated']==0 for r in rs),
            'exempt_no_better_option':sum(r['reason']=='no_strictly_better_expected_reduction_offered' for r in rs),
            'exempt_resolved_to_single_answer':sum(r['reason']=='answer_determined_afterward' for r in rs),
            'exempt_winning_guess':sum(r['reason']=='winning_guess' for r in rs)}
    return {**summarize(rows),'games':len(scores),
        'games_with_penalty':sum(g['penalty_count']>0 for g in scores),
        'mean_penalty':float(np.mean([g['low_information_penalty'] for g in scores])) if scores else None,
        'mean_training_reward':float(np.mean([g['training_reward'] for g in scores])) if scores else None,
        'by_turn':{str(t):summarize([r for r in rows if r['turn']==t]) for t in (4,5)}}
