"""Single-forward pointer, vocabulary-label and character readouts; no decoding."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import re
import torch
from torch import nn
from torch.nn import functional as F
from .environment import Decision

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
LORA_TARGETS = r".*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)"
ARCHITECTURES = {"char": "history-char-shared-mlp-v1", "pointer": "contextual-pointer-v1",
                 "lm_head": "vocabulary-label-lm-head-v1"}
POINTER_MARKERS = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>",
                   "<|box_end|>", "<|fim_suffix|>")


class CandidateHead(nn.Module):
    def __init__(self, hidden_size, width=256, char_size=32):
        super().__init__()
        self.history = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width), nn.GELU())
        self.characters = nn.Embedding(26, char_size)
        # Flatten the five positions in order; pooling letters would lose spelling.
        self.word = nn.Sequential(nn.Linear(5 * char_size, width), nn.GELU(), nn.LayerNorm(width))
        self.scorer = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, history, characters):
        h = self.history(history.float())[:, None, :]
        w = self.word(self.characters(characters).flatten(-2))
        h = h.expand(-1, w.shape[1], -1)
        return self.scorer(torch.cat([h, w, h * w], dim=-1)).squeeze(-1)


class SharedPolicy(nn.Module):
    def __init__(self, lm, tokenizer, head_width=256, model_spec=None):
        super().__init__()
        self.lm, self.tokenizer = lm, tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        core = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
        self.head = self.make_head(core.config.hidden_size, head_width)
        self.head.to(core.model.norm.weight.device)
        self.reference_head = None
        self._reference_active = False
        self.model_spec = model_spec or {}
        self.longest_prompt = 0

    def make_head(self, hidden_size, width):
        return CandidateHead(hidden_size, width)

    @property
    def backbone(self):
        core = self.lm.get_base_model() if hasattr(self.lm, "get_base_model") else self.lm
        return core.model

    def prompts(self, decisions):
        # Candidates/labels are intentionally absent from the language-model input.
        prompts = []
        for d in decisions:
            text = d.state_text()
            if getattr(self.tokenizer, "chat_template", None):
                text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
            prompts.append(text)
        return prompts

    def forward(self, decisions):
        if not decisions:
            raise ValueError("Empty decision batch")
        encoded = self.tokenizer(self.prompts(decisions), padding=True, return_tensors="pt",
                                 add_special_tokens=False)
        device = self.lm.get_input_embeddings().weight.device
        ids, mask = encoded.input_ids.to(device), encoded.attention_mask.to(device)
        self.longest_prompt = max(self.longest_prompt, ids.shape[1])
        position_ids = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
        # Exactly one backbone forward. No LM vocabulary projection and no generation.
        out = self.backbone(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                            use_cache=False, return_dict=True)
        hidden = out.last_hidden_state[:, -1]
        head = self.reference_head if self._reference_active else self.head
        head_device = next(head.parameters()).device
        chars = torch.tensor([[[ord(c) - ord("a") for c in w] for w in d.words]
                              for d in decisions], dtype=torch.long, device=head_device)
        return head(hidden.to(head_device), chars).float()

    def log_probs(self, decisions, temperature=1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        return F.log_softmax(self(decisions) / temperature, dim=-1)

    @torch.no_grad()
    def score(self, decisions, batch_size=8, temperature=1.0):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        was_training = self.training
        self.eval()
        try:
            rows = [self.log_probs(decisions[i:i+batch_size], temperature).cpu()
                    for i in range(0, len(decisions), batch_size)]
            return torch.cat(rows) if rows else torch.empty(0, 64)
        finally:
            self.train(was_training)

    def anchor_reference(self):
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        if "reference" not in self.lm.peft_config:
            self.lm.add_adapter("reference", deepcopy(self.lm.peft_config["policy"]))
        state = {k: v.detach().clone() for k, v in
                 get_peft_model_state_dict(self.lm, adapter_name="policy").items()}
        set_peft_model_state_dict(self.lm, state, adapter_name="reference")
        self.reference_head = deepcopy(self.head).requires_grad_(False)
        self.lm.set_adapter("policy")
        self._freeze_reference()

    def _freeze_reference(self):
        for name, p in self.lm.named_parameters():
            if ".reference." in name:
                p.requires_grad_(False)
        if self.reference_head is not None:
            self.reference_head.requires_grad_(False)

    @contextmanager
    def reference(self):
        if self.reference_head is None:
            raise RuntimeError("Reference has not been initialized")
        self.lm.set_adapter("reference")
        self._freeze_reference()
        self._reference_active = True
        try:
            yield
        finally:
            self._reference_active = False
            self.lm.set_adapter("policy")
            self._freeze_reference()

    def trainable_parameters(self):
        return [p for name, p in self.named_parameters()
                if p.requires_grad and ".reference." not in name and not name.startswith("reference_head.")]

    def drex_output(self, decision):
        p = self.score([decision])[0].exp().tolist()
        return {"next_guess": {f"O{i:02d}": x for i, x in enumerate(p)}}


class PointerHead(nn.Module):
    """Drex-style contextual option keys and final readout query, in float32."""
    def __init__(self, hidden_size, width=256):
        super().__init__()
        self.query = nn.Linear(hidden_size, width)
        self.key = nn.Linear(hidden_size, width)
        self.scale = width ** -0.5

    def forward(self, decision_hidden, option_hidden):
        q = self.query(decision_hidden.float())
        k = self.key(option_hidden.float())
        return (k * q[:, None, :]).sum(-1) * self.scale


class PointerPolicy(SharedPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ids = [self.tokenizer.encode(s, add_special_tokens=False) for s in POINTER_MARKERS]
        if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != 5:
            raise ValueError("Pointer boundaries must be distinct existing single tokens")
        self.marker_ids = tuple(x[0] for x in ids)

    def make_head(self, hidden_size, width):
        return PointerHead(hidden_size, width)

    def prompts(self, decisions):
        state, question, begin, end, decide = POINTER_MARKERS
        return [state + d.state_text() + question + "Which word should be guessed next?"
                + "".join(begin + w.upper() + end for w in d.words) + decide
                for d in decisions]

    def forward(self, decisions):
        if not decisions:
            raise ValueError("Empty decision batch")
        encoded = self.tokenizer(self.prompts(decisions), padding=True, return_tensors="pt",
                                 add_special_tokens=False)
        device = self.lm.get_input_embeddings().weight.device
        ids, mask = encoded.input_ids.to(device), encoded.attention_mask.to(device)
        endings = ids == self.marker_ids[3]
        if not bool((endings.sum(-1) == 64).all()) or not bool((ids[:, -1] == self.marker_ids[4]).all()):
            raise ValueError("Malformed pointer boundaries or truncated options")
        option_positions = endings.nonzero(as_tuple=False)[:, 1].reshape(len(decisions), 64)
        position_ids = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
        self.longest_prompt = max(self.longest_prompt, ids.shape[1])
        # One causal sequence per request: no cache prepass, option isolation,
        # generated labels, vocabulary projection, or per-option backbone calls.
        out = self.backbone(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                            use_cache=False, return_dict=True)
        hidden = out.last_hidden_state
        batch = torch.arange(len(decisions), device=hidden.device)[:, None]
        options = hidden[batch, option_positions.to(hidden.device)]
        head = self.reference_head if self._reference_active else self.head
        head_device = next(head.parameters()).device
        return head(hidden[:, -1].to(head_device), options.to(head_device)).float()


class LMHeadPolicy(SharedPolicy):
    """One backbone pass, then the pretrained LM output rows for option labels."""
    def make_head(self, hidden_size, width):
        from .lm_head_readout import select_labels, SelectedVocabularyRows
        self.labels, self.label_ids, self.readout_marker_ids = select_labels(self.tokenizer)
        core = self.lm.get_base_model() if hasattr(self.lm, 'get_base_model') else self.lm
        return SelectedVocabularyRows(core.get_output_embeddings(), self.label_ids)

    def prompts(self, decisions):
        from .lm_head_readout import prompt
        return [prompt(self.tokenizer, decision, self.labels) for decision in decisions]

    def forward(self, decisions):
        if not decisions:
            raise ValueError('Empty decision batch')
        encoded = self.tokenizer(self.prompts(decisions), padding=True, return_tensors='pt',
                                 add_special_tokens=False)
        device = self.lm.get_input_embeddings().weight.device
        ids, mask = encoded.input_ids.to(device), encoded.attention_mask.to(device)
        marker = torch.tensor(self.readout_marker_ids, device=device)
        if not bool((ids[:, -len(marker):] == marker).all()) or not bool(mask[:, -1].all()):
            raise ValueError('LM-head readout marker is missing or truncated')
        self.longest_prompt = max(self.longest_prompt, ids.shape[1])
        positions = (mask.long().cumsum(-1)-1).masked_fill(mask == 0, 0)
        out = self.backbone(input_ids=ids, attention_mask=mask, position_ids=positions,
                            use_cache=False, return_dict=True)
        head = self.reference_head if self._reference_active else self.head
        return head(out.last_hidden_state[:, -1].to(head.weight.device))


def explicit_device_map(config, placement):
    if placement == "cpu":
        return {"": "cpu"}
    if placement == "cuda:0":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        return {"": 0}
    if placement != "two_gpu" or torch.cuda.device_count() < 2:
        raise RuntimeError("two_gpu requires two visible CUDA GPUs; use cpu only for tiny tests")
    # Single-process GPU model parallelism. No CPU/disk offload or inference auto-map.
    n = config.num_hidden_layers
    mapping = {"model.embed_tokens": 0, "model.rotary_emb": 0,
               "model.norm": 1, "lm_head": 1}
    mapping.update({f"model.layers.{i}": 0 if i < n // 2 else 1 for i in range(n)})
    return mapping


def build_policy(model_id=None, placement="two_gpu", dtype="bfloat16", rank=16,
                 alpha=32, head_width=256, grad_checkpointing=True, checkpoint=None,
                 local_files_only=True, head_seed=None, architecture=None, revision=None):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    from .contracts import require_fingerprint
    meta = None
    if checkpoint:
        checkpoint = Path(checkpoint)
        meta = json.loads((checkpoint / "metadata.json").read_text())
        require_fingerprint(meta["contract"])
        spec = meta["model_spec"]
        model_id = model_id or spec["model_id"]
        rank, alpha, head_width = (spec[k] for k in ["rank", "alpha", "head_width"])
        saved_arch = next((k for k, v in ARCHITECTURES.items() if v == spec["architecture"]), None)
        if saved_arch is None or (architecture is not None and architecture != saved_arch):
            raise ValueError("Checkpoint readout architecture does not match requested architecture")
        architecture = saved_arch
    architecture = architecture or "char"
    model_id = model_id or MODEL_ID
    revision = revision or (MODEL_REVISION if model_id == MODEL_ID else None)
    source = {"local_files_only": local_files_only, "revision": revision}
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported readout architecture: {architecture}")
    config = AutoConfig.from_pretrained(model_id, **source)
    model_class = AutoModelForCausalLM
    if config.model_type == "qwen3_5_moe":
        from transformers import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM
        config = Qwen3_5MoeTextConfig(**config.text_config.to_dict())
        model_class = Qwen3_5MoeForCausalLM
    elif config.model_type == "qwen3_5_moe_text":
        from transformers import Qwen3_5MoeForCausalLM
        model_class = Qwen3_5MoeForCausalLM
    config.use_cache = False
    model, loading = model_class.from_pretrained(
        model_id, config=config, dtype=getattr(torch, dtype),
        device_map=explicit_device_map(config, placement), attn_implementation="sdpa",
        **source, output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise RuntimeError(f"Refusing partially initialized checkpoint: {loading}")
    if placement != "cpu" and any(p.device.type != "cuda" for p in model.parameters()):
        raise RuntimeError("CPU/disk offload is not supported by this training runner")
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint / "adapters/policy",
                                          adapter_name="policy", is_trainable=True)
    else:
        cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                         target_modules=LORA_TARGETS, bias="none")
        model = get_peft_model(model, cfg, adapter_name="policy")
    if grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    tok = AutoTokenizer.from_pretrained(model_id, **source)
    spec = {"model_id": str(model_id), "rank": rank, "alpha": alpha, "head_width": head_width,
            "lora_targets": LORA_TARGETS, "architecture": ARCHITECTURES[architecture]}
    policy_class = {'char': SharedPolicy, 'pointer': PointerPolicy, 'lm_head': LMHeadPolicy}[architecture]
    if head_seed is None:
        policy = policy_class(model, tok, head_width, spec)
    else:
        # LoRA rank changes RNG consumption; initialize the common head independently.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(head_seed)
            policy = policy_class(model, tok, head_width, spec)
        policy.model_spec["head_seed"] = head_seed
    if checkpoint:
        policy.model_spec.update(meta["model_spec"])
        policy.head.load_state_dict(torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True))
        if meta.get("has_reference"):
            model.load_adapter(checkpoint / "adapters/reference", adapter_name="reference", is_trainable=False)
            policy.reference_head = deepcopy(policy.head).requires_grad_(False)
            policy.reference_head.load_state_dict(torch.load(checkpoint / "reference_head.pt", map_location="cpu", weights_only=True))
            model.set_adapter("policy")
            policy._freeze_reference()
    coverage = [n for n, _ in model.named_modules() if n.endswith(".lora_A")]
    policy.model_spec["adapted_module_count"] = len(coverage)
    if architecture == "pointer":
        marker_ids = dict(zip(POINTER_MARKERS, policy.marker_ids))
        if checkpoint and policy.model_spec.get("pointer_marker_ids") != marker_ids:
            raise ValueError("Checkpoint pointer boundary tokenizer differs")
        policy.model_spec["pointer_marker_ids"] = marker_ids
    if architecture == 'lm_head':
        from .lm_head_readout import MARKER
        label_spec = {'lm_head_labels': policy.labels, 'lm_head_label_ids': policy.label_ids,
                      'lm_head_marker': MARKER, 'lm_head_marker_ids': policy.readout_marker_ids}
        if checkpoint and any(meta['model_spec'].get(k) != v for k, v in label_spec.items()):
            raise ValueError('LM-head checkpoint label mapping or marker differs from tokenizer')
        if policy.head.token_ids.tolist() != policy.label_ids:
            raise ValueError('Saved vocabulary rows do not match their labels')
        policy.model_spec.update(label_spec, lm_head_trainable_rows=64,
            lm_head_initialization='selected-pretrained-vocabulary-rows',
            lm_head_compute_dtype=str(policy.head.compute_dtype))
    return policy
