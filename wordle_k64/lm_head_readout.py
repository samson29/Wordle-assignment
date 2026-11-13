"""Selected pretrained vocabulary rows at an explicit option-label marker."""
import torch
from torch import nn
from torch.nn import functional as F

MARKER = 'Next option:\n'
LABEL_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/@#$%&*=?'
RULE = 'Reply with exactly one option label, without explanation or the word.'


def select_labels(tokenizer):
    prefix = tokenizer.encode(MARKER, add_special_tokens=False)
    labels, token_ids = [], []
    special = set(tokenizer.all_special_ids)
    for label in LABEL_ALPHABET:
        ids = tokenizer.encode(MARKER + label, add_special_tokens=False)
        if (len(ids) == len(prefix)+1 and ids[:-1] == prefix
                and ids[-1] not in token_ids and ids[-1] not in special):
            labels.append(label); token_ids.append(ids[-1])
        if len(labels) == 64:
            break
    if len(labels) != 64:
        raise ValueError('Need 64 distinct single-token label continuations at the marker')
    return labels, token_ids, prefix


def prompt(tokenizer, decision, labels):
    text = (decision.state_text() + '\nWhich word should be guessed next?\nOptions:\n'
            + '\n'.join(f'{label}: {word.upper()}' for label, word in zip(labels, decision.words))
            + '\n' + RULE)
    if getattr(tokenizer, 'chat_template', None):
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text + MARKER


def verify_continuations(tokenizer, text, labels, token_ids):
    """Run in preflight on actual complete prompts, not just isolated label strings."""
    before = tokenizer.encode(text, add_special_tokens=False)
    for label, token_id in zip(labels, token_ids):
        if tokenizer.encode(text + label, add_special_tokens=False) != before + [token_id]:
            raise ValueError(f'Label {label!r} is not the expected single-token continuation')


class SelectedVocabularyRows(nn.Module):
    """Train only 64 output rows; leave the frozen full vocabulary matrix untouched.

    This is a sparse-row adaptation of the pretrained LM head, not a randomly
    initialized classifier. FP32 master parameters are projected in the original
    LM-head dtype so step-zero scoring follows the native vocabulary projection.
    """
    def __init__(self, pretrained_head, token_ids):
        super().__init__()
        ids = torch.tensor(token_ids, dtype=torch.long, device=pretrained_head.weight.device)
        self.register_buffer('token_ids', ids)
        self.compute_dtype = pretrained_head.weight.dtype
        self.weight = nn.Parameter(pretrained_head.weight.detach().index_select(0, ids).float().clone())
        bias = getattr(pretrained_head, 'bias', None)
        self.bias = None if bias is None else nn.Parameter(bias.detach().index_select(0, ids).float().clone())

    def forward(self, hidden):
        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        return F.linear(hidden.to(self.compute_dtype), self.weight.to(self.compute_dtype), bias).float()
