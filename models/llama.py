import math
import logging
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from transformers.models.llama.modeling_llama import (
    LlamaConfig, 
    LlamaRMSNorm, 
    LlamaRotaryEmbedding, 
    LlamaModel, 
    LlamaForCausalLM, 
    apply_rotary_pos_emb, 
    repeat_kv
)
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache

logger = logging.getLogger(__name__)


class CustomLinear(nn.Module):
    def __init__(self, input_dim, output_dim, vector_dim, bias=False):
        super().__init__()
        assert input_dim * output_dim % vector_dim == 0
        index_length = input_dim * output_dim // vector_dim

        self.index = nn.Parameter(torch.randint(high=index_length, size=(index_length,), dtype=torch.int32), requires_grad=False)
        self.output_dim = output_dim  
        self.input_dim = input_dim
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x, vector_bank):
        vector = vector_bank[self.index] 
        weight = vector.view(self.input_dim, self.output_dim)
        if self.bias is not None:
            out = torch.matmul(x, weight) + self.bias
        else:
            out = torch.matmul(x, weight)
        return out

    
class CustomLlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.vector_dim = config.vector_bank_dim
        
        self.gate_proj = CustomLinear(self.hidden_size, self.intermediate_size, self.vector_dim, bias=config.mlp_bias)
        self.up_proj = CustomLinear(self.hidden_size, self.intermediate_size, self.vector_dim, bias=config.mlp_bias)
        self.down_proj = CustomLinear(self.intermediate_size, self.hidden_size, self.vector_dim, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]
        
    def forward(self, x, vector_bank):
        down_proj = self.down_proj(
            self.act_fn(
                self.gate_proj(x, vector_bank)) * self.up_proj(x, vector_bank),
            vector_bank
        )
        return down_proj
    
    
class CustomLlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.vector_dim = config.vector_bank_dim
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = CustomLinear(self.hidden_size, self.num_heads * self.head_dim, self.vector_dim, bias=config.attention_bias)
        self.k_proj = CustomLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, self.vector_dim, bias=config.attention_bias)
        self.v_proj = CustomLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, self.vector_dim, bias=config.attention_bias)
        self.o_proj = CustomLinear(self.hidden_size, self.hidden_size, self.vector_dim, bias=config.attention_bias)

        # TODO (joao): remove in v4.45 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        vector_bank: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states, vector_bank)
        key_states = self.k_proj(hidden_states, vector_bank)
        value_states = self.v_proj(hidden_states, vector_bank)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output, vector_bank)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    
class CustomLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = CustomLlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = CustomLlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vector_bank = nn.Parameter(torch.rand(config.vector_bank_length, config.vector_bank_dim), requires_grad=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            vector_bank=self.vector_bank,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, self.vector_bank)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    
class CustomLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [CustomLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
class CustomLlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModel(config)