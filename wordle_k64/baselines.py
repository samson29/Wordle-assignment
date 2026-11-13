"""Read-only comparison policies. Features are computed from public histories only."""
import numpy as np
import torch
from collections import Counter
from .environment import seed_for


class SolverPolicy:
    def __init__(self, engine, sampler, mode):
        from .planner import ChallengePlanner
        self.engine, self.sampler, self.mode = engine, sampler, mode
        self.planner = ChallengePlanner(engine, sampler) if mode == "planner" else None

    def score(self, decisions, batch_size=8, temperature=1.0):
        rows = []
        for d in decisions:
            s = self.engine.state(d.history)
            ids = tuple(self.engine.gid[w] for w in d.words)
            possible = {self.engine.answer_to_guess[a] for a in s.answers}
            if self.mode == "planner":
                chosen, _ = self.planner.choose(s, ids)
            elif self.mode == "entropy":
                chosen = min(ids, key=lambda g: (-(g in possible), self.engine.guesses[g])) if s.remaining_turns == 1 else min(
                    ids, key=lambda g: (-round(self.engine.entropy(s, g), 12), -(g in possible), self.engine.guesses[g]))
            elif self.mode == "consistent":
                # Root9 has no consistent opening option. Apply the same public
                # letter-frequency heuristic to all offered words in that case.
                choices = [g for g in ids if g in possible] or list(ids)
                counts = Counter(c for a in s.answers for c in set(self.engine.answers[a]))
                chosen = min(choices, key=lambda g: (-sum(counts[c] for c in set(self.engine.guesses[g])),
                                                    self.engine.guesses[g]))
            elif self.mode == "random":
                rows.append(torch.full((64,), -np.log(64), dtype=torch.float32))
                continue
            else:
                raise ValueError(self.mode)
            row = torch.full((64,), -torch.inf)
            row[ids.index(chosen)] = 0
            rows.append(row)
        return torch.stack(rows)


class NativeLabelPolicy:
    """Frozen native CausalLM; one forward, select logits for 64 vocabulary labels."""
    def __init__(self, model_id=None, placement='two_gpu',
                 dtype='bfloat16', local_files_only=True, revision=None):
        import torch
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM
        from .model import explicit_device_map, MODEL_ID, MODEL_REVISION
        from wordle_k64.lm_head_readout import select_labels
        torch.manual_seed(20261112)
        torch.set_num_threads(8)
        model_id = model_id or MODEL_ID
        revision = revision or (MODEL_REVISION if model_id == MODEL_ID else None)
        source = {'local_files_only': local_files_only, 'revision': revision}
        config = AutoConfig.from_pretrained(model_id, **source)
        if config.model_type == 'qwen3_5_moe':
            config = Qwen3_5MoeTextConfig(**config.text_config.to_dict())
        model_class = Qwen3_5MoeForCausalLM if config.model_type == 'qwen3_5_moe_text' else AutoModelForCausalLM
        config.use_cache = False
        device_map = explicit_device_map(config, placement)
        self.lm, self.loading = model_class.from_pretrained(
            model_id, config=config, dtype=getattr(torch, dtype), device_map=device_map,
            attn_implementation='sdpa', **source, output_loading_info=True)
        if self.loading.get('missing_keys') or self.loading.get('mismatched_keys') or self.loading.get('error_msgs'):
            raise RuntimeError(f'Partially initialized model: {self.loading}')
        self.lm.eval().requires_grad_(False)
        assert all(not p.requires_grad for p in self.lm.parameters())
        assert not hasattr(self.lm, 'peft_config')
        assert not any('lora' in n.lower() or 'adapter' in n.lower() for n, _ in self.lm.named_parameters())
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **source)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.labels, self.label_ids, self.marker_ids = select_labels(self.tokenizer)
        self.label_tensor = torch.tensor(self.label_ids, device=self.lm.get_output_embeddings().weight.device)
        self.longest_prompt = 0

    def prompts(self, decisions):
        from wordle_k64.lm_head_readout import prompt
        return [prompt(self.tokenizer, d, self.labels) for d in decisions]

    def native_logits(self, decisions):
        import torch
        texts = self.prompts(decisions)
        encoded = self.tokenizer(texts, padding=True, return_tensors='pt', add_special_tokens=False)
        device = self.lm.get_input_embeddings().weight.device
        ids = encoded.input_ids.to(device)
        mask = encoded.attention_mask.to(device)
        marker = torch.tensor(self.marker_ids, device=device)
        assert bool((ids[:, -len(marker):] == marker).all())
        assert bool(mask[:, -1].all())
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
        self.longest_prompt = max(self.longest_prompt, ids.shape[1])
        # Match the SFT marker-hidden-state readout path, using the untouched native
        # full vocabulary head. No copied, initialized, or adapted output rows.
        output = self.lm.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                               use_cache=False, return_dict=True)
        head = self.lm.get_output_embeddings()
        logits = head(output.last_hidden_state[:, -1].to(head.weight.device))
        return logits.index_select(-1, self.label_tensor.to(logits.device)).float()

    def score(self, decisions, batch_size=8, temperature=1.0):
        import torch
        assert batch_size > 0 and temperature > 0
        with torch.inference_mode():
            parts = [torch.log_softmax(self.native_logits(decisions[i:i+batch_size]) / temperature, -1).cpu()
                     for i in range(0, len(decisions), batch_size)]
        return torch.cat(parts) if parts else torch.empty(0, 64)

    def frozen_parameter_versions(self):
        return {n: p._version for n, p in self.lm.named_parameters()}
