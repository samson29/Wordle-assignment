"""Separate adapter/head rates and word-aligned permutation regularization."""
import numpy as np
import torch


def optimizer_groups(policy, adapter_lr, head_lr):
    """Cover every trainable policy tensor once; frozen references stay excluded."""
    trainable = policy.trainable_parameters()
    head_ids = {id(p) for p in policy.head.parameters()}
    backbone = [p for p in trainable if id(p) not in head_ids]
    head = [p for p in trainable if id(p) in head_ids]
    if not backbone or not head or len({id(p) for p in backbone + head}) != len(trainable):
        raise ValueError('Invalid adapter/head parameter partition')
    return [{'name': 'backbone_lora', 'params': backbone, 'lr': adapter_lr},
            {'name': 'pointer_head', 'params': head, 'lr': head_lr}]


def permutation_batch(decisions, rng):
    """Return shuffled public inputs and inverse indices back to original words."""
    rows = [d.shuffled(rng) for d in decisions]
    return [d for d, _ in rows], np.stack([perm.argsort() for _, perm in rows])


def align_log_probs(log_probs, inverse):
    indices = torch.as_tensor(inverse, dtype=torch.long, device=log_probs.device)
    return log_probs.gather(-1, indices)


def symmetric_kl(log_p, log_q):
    """Per-decision symmetric KL, differentiable through BOTH distributions."""
    log_p, log_q = log_p.float(), log_q.float()
    return 0.5 * ((log_p.exp() - log_q.exp()) * (log_p - log_q)).sum(-1)
