from copy import deepcopy
import numpy as np
import torch
from wordle_k64.environment import Game
from wordle_k64.model import build_policy
from wordle_k64.checkpoint import seed_everything, save_checkpoint, resume_training
from wordle_k64.sl_objectives import optimizer_groups, permutation_batch, align_log_probs, symmetric_kl
from _common import warmup


def test_word_alignment_and_consistency_gradients(env):
    e, s = env
    d = Game(e, s, 'crane', 23).decision()
    shuffled, inverse = permutation_batch([d], np.random.default_rng(19))
    assert shuffled[0].history == d.history and set(shuffled[0].words) == set(d.words)
    logits = lambda ds: torch.tensor([[sum((i+1)*ord(c) for i,c in enumerate(w))/100
                                       for w in d.words] for d in ds])
    p = logits([d]).log_softmax(-1)
    q = align_log_probs(logits(shuffled).log_softmax(-1), inverse)
    assert torch.allclose(p, q, atol=1e-6)
    assert symmetric_kl(p,q).abs().max() < 1e-10
    a = torch.tensor([[2., 0., -1.]], requires_grad=True)
    b = torch.tensor([[-1., 0., 2.]], requires_grad=True)
    ap,bp = a.log_softmax(-1),b.log_softmax(-1)
    value = symmetric_kl(ap,bp).sum()
    assert value > 0 and torch.allclose(value,symmetric_kl(bp,ap).sum())
    value.backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0


def test_separate_rates_resume_and_pair_gradients(tiny, env, tmp_path):
    e,s = env
    rng = seed_everything(44)
    paired_rng = np.random.default_rng(123)
    p = build_policy(tiny, placement='cpu', dtype='float32', rank=4, alpha=8,
                     head_width=32, head_seed=91, architecture='pointer')
    p.anchor_reference()
    groups = optimizer_groups(p, 1e-4, 2e-4)
    wanted = {id(x) for x in p.trainable_parameters()}
    assert {id(x) for g in groups for x in g['params']} == wanted
    assert sum(len(g['params']) for g in groups) == len(wanted)
    opt = torch.optim.AdamW(groups, weight_decay=0)
    sched = warmup(opt,20)
    d = Game(e,s,'crane',91).decision()
    cfg = {'head_lr':2e-4, 'lr':1e-4, 'consistency_coef':0.05}

    def update(policy, optimizer, scheduler, generator):
        policy.train(); optimizer.zero_grad()
        shuffled,inverse = permutation_batch([d],generator)
        lp = policy.log_probs([d]);lq = align_log_probs(policy.log_probs(shuffled),inverse)
        loss = -lp[0,0] + 0.05*symmetric_kl(lp,lq).mean()
        loss.backward()
        assert policy.head.query.weight.grad.abs().sum() > 0
        assert policy.head.key.weight.grad.abs().sum() > 0
        assert any(x.grad is not None and x.grad.abs().sum() > 0 for n,x in policy.named_parameters() if '.policy.' in n)
        assert abs(optimizer.param_groups[1]['lr']/optimizer.param_groups[0]['lr']-2) < 1e-12
        optimizer.step();scheduler.step()
        return float(loss.detach())

    update(p,opt,sched,paired_rng)
    path = tmp_path/'resume'
    progress = {'step':1,'consistency_rng_state':deepcopy(paired_rng.bit_generator.state)}
    save_checkpoint(p,path,optimizer=opt,scheduler=sched,rng=rng,progress=progress,training_config=cfg)
    expected_loss = update(p,opt,sched,paired_rng)
    expected_probs = p.score([d])
    loaded = build_policy(checkpoint=path,placement='cpu',dtype='float32')
    opt2 = torch.optim.AdamW(optimizer_groups(loaded,1e-4,2e-4),weight_decay=0)
    sched2 = warmup(opt2,20)
    restored = resume_training(path,opt2,sched2,rng,cfg)
    resumed_pairs = np.random.default_rng(999)
    resumed_pairs.bit_generator.state = restored['consistency_rng_state']
    actual_loss = update(loaded,opt2,sched2,resumed_pairs)
    assert abs(actual_loss-expected_loss) < 1e-6
    assert torch.allclose(loaded.score([d]),expected_probs,atol=1e-6)


def test_equal_rate_groups_match_original_optimizer():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.adapter=torch.nn.Linear(2,2);self.head=torch.nn.Linear(2,1)
        def trainable_parameters(self):return list(self.parameters())
        def forward(self,x):return self.head(self.adapter(x))
    torch.manual_seed(73);first=Toy();second=deepcopy(first)
    a=torch.optim.AdamW(first.parameters(),lr=1e-4,weight_decay=0)
    b=torch.optim.AdamW(optimizer_groups(second,1e-4,1e-4),weight_decay=0)
    for _ in range(3):
        for p,opt in [(first,a),(second,b)]:
            opt.zero_grad();p(torch.ones(2,2)).square().mean().backward();opt.step()
    assert all(torch.equal(x,y) for x,y in zip(first.parameters(),second.parameters()))
