import logging
import torch
import torch.nn as nn
from typing import Optional
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, 
    LlamaSdpaAttention, 
    LlamaMLP, 
    LlamaConfig, 
    LlamaModel, 
    LlamaForCausalLM
)

logger = logging.getLogger(__name__)


class CustomLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_clusters: int, cluster_dim: int, bias: bool = True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.deficiency = out_features % cluster_dim
        if self.deficiency > 0:
            self.deficiency = cluster_dim - self.deficiency

        index_length = in_features * (out_features + self.deficiency) // cluster_dim
        self.cluster = nn.Parameter(torch.empty((num_clusters, cluster_dim), **factory_kwargs))
        index = torch.empty((index_length,), dtype=torch.uint16, device=device) 
        self.register_buffer('index', index)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        vectors = self.cluster[self.index.to(torch.int32)]
        if self.deficiency > 0:
            weight = vectors.view(self.in_features, -1)[:, :-self.deficiency]
        else:
            weight = vectors.view(self.in_features, -1)
            
        if self.bias is not None:
            out = torch.matmul(x, weight) + self.bias
        else:
            out = torch.matmul(x, weight)
        return out

class CustomLlamaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_clusters = config.num_clusters
        self.cluster_dim = config.cluster_dim
        self.gate_proj = CustomLinear(self.hidden_size, self.intermediate_size, self.num_clusters, self.cluster_dim, bias=config.mlp_bias)
        self.up_proj = CustomLinear(self.hidden_size, self.intermediate_size, self.num_clusters, self.cluster_dim, bias=config.mlp_bias)
        self.down_proj = CustomLinear(self.intermediate_size, self.hidden_size, self.num_clusters, self.cluster_dim, bias=config.mlp_bias)
        
class CustomLlamaSdpaAttention(LlamaSdpaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_clusters = config.num_clusters
        self.cluster_dim = config.cluster_dim
        self.q_proj = CustomLinear(self.hidden_size, self.num_heads * self.head_dim, self.num_clusters, self.cluster_dim, bias=config.attention_bias)
        self.k_proj = CustomLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, self.num_clusters, self.cluster_dim, bias=config.attention_bias)
        self.v_proj = CustomLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, self.num_clusters, self.cluster_dim, bias=config.attention_bias)
        self.o_proj = CustomLinear(self.hidden_size, self.hidden_size, self.num_clusters, self.cluster_dim, bias=config.attention_bias)

class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = CustomLlamaSdpaAttention(config=config, layer_idx=layer_idx)
        self.mlp = CustomLlamaMLP(config)

class CustomLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [CustomLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModel(config)