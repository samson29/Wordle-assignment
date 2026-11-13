"""Create a tiny random model for mechanics tests; it is NOT a useful Wordle model."""
import argparse
import string


def create(path, kind="llama"):
    import torch
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    chars = sorted(set(string.printable) - set("\t\r\x0b\x0c"))
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, **{c: i + 3 for i, c in enumerate(chars)}}
    # Tiny mechanics fixture only: the real Qwen vocabulary already has these.
    from wordle_k64.model import POINTER_MARKERS
    vocab.update({s: len(vocab) + i for i, s in enumerate(POINTER_MARKERS)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    tok.add_special_tokens({'additional_special_tokens': list(POINTER_MARKERS)})
    tok.save_pretrained(path)
    torch.manual_seed(42)
    if kind == "qwen":
        from transformers import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM
        cfg = Qwen3_5MoeTextConfig(vocab_size=len(vocab), hidden_size=64, num_hidden_layers=4,
              num_attention_heads=2, num_key_value_heads=1, head_dim=32,
              linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
              linear_num_value_heads=2, num_experts=4, num_experts_per_tok=2,
              moe_intermediate_size=32, shared_expert_intermediate_size=32,
              rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                               "partial_rotary_factor": 0.5, "mrope_section": [2, 3, 3]},
              pad_token_id=0, use_cache=False)
        model = Qwen3_5MoeForCausalLM(cfg)
    else:
        cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                          max_position_embeddings=8192, pad_token_id=0, use_cache=False)
        model = LlamaForCausalLM(cfg)
    model.save_pretrained(path)
    return str(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/tiny_model")
    p.add_argument("--kind", choices=["llama", "qwen"], default="llama")
    a = p.parse_args()
    print(create(a.out, a.kind))
