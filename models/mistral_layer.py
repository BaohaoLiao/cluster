import logging
import torch
import torch.nn as nn
from typing import Optional
from transformers.models.mistral.modeling_mistral import (
    MistralMLP,
    MistralSdpaAttention,
    MistralConfig,
    MistralDecoderLayer,
    MistralModel,
    MistralForCausalLM,
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
        index = torch.empty((index_length,), dtype=torch.int32, device=device) # save_pretraied doesn't support uint16
        self.register_buffer('index', index)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        vectors = self.cluster[self.index]
        if self.deficiency > 0:
            weight = vectors.view(self.in_features, -1)[:, :-self.deficiency]
        else:
            weight = vectors.view(self.in_features, -1)
            
        if self.bias is not None:
            out = torch.matmul(x, weight) + self.bias
        else:
            out = torch.matmul(x, weight)
        return out

class CustomMistralMLP(MistralMLP):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_clusters = config.num_clusters
        self.cluster_dim = config.cluster_dim
        self.gate_proj = CustomLinear(self.hidden_size, self.intermediate_size, self.num_clusters, self.cluster_dim, bias=False)
        self.up_proj = CustomLinear(self.hidden_size, self.intermediate_size, self.num_clusters, self.cluster_dim, bias=False)
        self.down_proj = CustomLinear(self.intermediate_size, self.hidden_size, self.num_clusters, self.cluster_dim, bias=False)
        
class CustomMistralSdpaAttention(MistralSdpaAttention):
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_clusters = config.num_clusters
        self.cluster_dim = config.cluster_dim
        self.q_proj = CustomLinear(self.hidden_size, self.num_heads * self.head_dim, self.num_clusters, self.cluster_dim, bias=False)
        self.k_proj = CustomLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, self.num_clusters, self.cluster_dim, bias=False)
        self.v_proj = CustomLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, self.num_clusters, self.cluster_dim, bias=False)
        self.o_proj = CustomLinear(self.num_heads * self.head_dim, self.hidden_size, self.num_clusters, self.cluster_dim, bias=False)

class CustomMistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = CustomMistralSdpaAttention(config=config, layer_idx=layer_idx)
        self.mlp = CustomMistralMLP(config)

class CustomMistralModel(MistralModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [CustomMistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

class CustomMistralForCausalLM(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomMistralModel(config)