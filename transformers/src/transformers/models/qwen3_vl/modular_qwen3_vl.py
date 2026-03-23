# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen3-VL model."""

# from typing import Callable, Optional, Union # gyn 
from typing import Any, Dict, List, Optional, Tuple, Union # gyn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint   # gyn
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn.utils.rnn import pad_sequence

# gyn 添加flow matching相关的导入
import math
from dataclasses import dataclass
try:
    from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
    from torchcfm.utils import *
    from torchcfm.models.models import *
except ImportError:
    print("Warning: torchcfm not found. Please install it for flow matching functionality.")
    ConditionalFlowMatcher = None


from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update, rope_config_validation
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import ProcessingKwargs, Unpack, VideosKwargs
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_torchdynamo_compiling, logging
from ...utils.generic import check_model_inputs
from ...video_utils import VideoInput
from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLVisionBlock,
)
from ..qwen2_vl.modeling_qwen2_vl import (
    PatchEmbed,
    Qwen2VLModelOutputWithPast,
    Qwen2VLPreTrainedModel,
    TransformersKwargs,
    VisionAttention,
    VisionRotaryEmbedding,
)
from ..qwen2_vl.processing_qwen2_vl import Qwen2VLImagesKwargs, Qwen2VLProcessor
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3Model,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


logger = logging.get_logger(__name__)


class Qwen3VLVisionConfig(PretrainedConfig):
    model_type = "qwen3_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24],
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = deepstack_visual_indexes


class Qwen3VLTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3VLTextModel`]. It is used to instantiate a
    Qwen3-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3-VL-4B-Instruct [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen3VL model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen3VLModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        head_dim (`int`, *optional*, defaults to 128):
            The dimension of the head. If not specified, will default to `hidden_size // num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 128000):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 5000000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen3VLTextModel, Qwen3VLTextConfig

    >>> # Initializing a Qwen3VL style configuration
    >>> configuration = Qwen3VLTextConfig()

    >>> # Initializing a model from the Qwen3-VL-7B style configuration
    >>> model = Qwen3VLTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_vl_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=5000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        rope_config_validation(self, ignore_keys={"mrope_section", "mrope_interleaved"})

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3VLConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3VLModel`]. It is used to instantiate a
    Qwen3-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3-VL-4B-Instruct [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen3VLTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Qwen3VLVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The start token index to encode the image prompt.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The end token index to encode the image prompt.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the word embeddings.

    ```python
    >>> from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig

    >>> # Initializing a Qwen3-VL style configuration
    >>> configuration = Qwen3VLConfig()

    >>> # Initializing a model from the Qwen3-VL-4B style configuration
    >>> model = Qwen3VLForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_vl"
    sub_configs = {"vision_config": Qwen3VLVisionConfig, "text_config": Qwen3VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionPatchEmbed(PatchEmbed):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)


class Qwen3VLVisionRotaryEmbedding(VisionRotaryEmbedding):
    pass


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class Qwen3VLVisionAttention(VisionAttention):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size


class Qwen3VLVisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)


class Qwen3VLTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3VLTextConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", "default")
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3VLTextAttention(Qwen3Attention):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3VLTextDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.attention_type

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        return super().forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )


class Qwen3VLModelOutputWithPast(Qwen2VLModelOutputWithPast):
    pass


class Qwen3VLPreTrainedModel(Qwen2VLPreTrainedModel):
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]
    _can_record_outputs = {
        "hidden_states": Qwen3VLTextDecoderLayer,
        "attentions": Qwen3VLTextAttention,
    }


class Qwen3VLVisionModel(Qwen3VLPreTrainedModel):
    config: Qwen3VLVisionConfig
    _no_split_modules = ["Qwen3VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            config=config,
        )

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists


@auto_docstring(
    custom_intro=(
        "Text part of Qwen3VL, "
        "not a pure text-only model, as DeepStack integrates visual features into the early hidden states."
    )
)
class Qwen3VLTextModel(Qwen3VLPreTrainedModel, Qwen3Model):
    config: Qwen3VLTextConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__(config)
        del self.has_sliding_layers

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class Qwen3VLModel(Qwen2_5_VLModel):
    config: Qwen3VLConfig
    _checkpoint_conversion_mapping = {}
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    @auto_docstring
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


class Qwen3VLCausalLMOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    pass


# gyn waypoints encoder
class MultiLayerEmbedding(nn.Module):
    def __init__(self, input_dim=2, embedding_dim=1536, hidden_dim=512):
        super(MultiLayerEmbedding, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, embedding_dim)
        self.activation = nn.ReLU()

    def forward(self, coords):
        if coords.dim() != 3 or coords.size(-1) != 2:
            raise ValueError(f"Expected coords of shape (B, N, 2), but got {coords.shape}")
        B = coords.shape[0]
        coords = coords.view(B, -1)  # Flatten the coordinates
        x = self.activation(self.layer1(coords))
        enc = self.layer2(x)
        return enc

# gyn action former        
class HeadBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, q, k, v):
        attn_out, _ = self.mha(q, k, v)
        out1 = self.norm1(q + attn_out)
        ffn_out = self.ffn(out1)
        out2 = self.norm2(out1 + ffn_out)
        return out2

class ActionFormerHead(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            HeadBlock(hidden_dim, num_heads) for _ in range(num_blocks)
        ])
    def forward(self, q, k, v):
        for block in self.blocks:
            q = block(q, k, v)
        return q

    def _initialize_weights(self, module=None):
        if module is None:
            module = self
        for m in module.modules():
            # 常见初始化规则
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.MultiheadAttention):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb

class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x

class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=4,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def _init_weights(self):
        print('initing weights of NoisePredictionModel...')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output

class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=2,
        action_chunk=5,
        num_diffusion_steps_train=100,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*action_dim, hidden_dim=hidden_dim, action_dim=action_dim
        )
        self.num_diffusion_steps_train = num_diffusion_steps_train
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps_train, beta_schedule="squaredcos_cap_v2")
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)
        self.action_chunk = action_chunk
    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, self.action_chunk, self.action_dim), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, self.action_chunk, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# class TransformerForDiffusion(ModuleAttrMixin):
class TransformerForDiffusion(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 horizon: int,
                 n_obs_steps: int = None,
                 cond_dim: int = 0,
                 n_layer: int = 12,
                 n_head: int = 12,
                 n_emb: int = 768,
                 p_drop_emb: float = 0.1,
                 p_drop_attn: float = 0.1,
                 causal_attn: bool = False,
                 time_as_cond: bool = True,
                 obs_as_cond: bool = False,
                 n_cond_layers: int = 0
                 ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None

        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True  # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s - 1)  # add one dimension since time is the first token in cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
        self.model_dtype = next(self.parameters()).dtype

    def _initialize_weights(self, module=None):
        # 让它调用你已有的 _init_weights
        if module is None:
            module = self
        return self._init_weights(module)

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        SinusoidalPosEmb,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self,
                             learning_rate: float = 1e-4,
                             weight_decay: float = 1e-3,
                             betas: Tuple[float, float] = (0.9, 0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                cond: Optional[torch.Tensor] = None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        # print('velocity_predictor sample.dtype', sample.dtype)
        # print('velocity_predictor input_emb.weight dtype', self.input_emb.weight.dtype)

        # 确保所有输入张量具有相同的数据类型
        model_dtype = self.input_emb.weight.dtype
        sample = sample.to(model_dtype)
        if cond is not None: # torch.Size([1, 1536])
            cond = cond.to(model_dtype)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # process input
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # BERT
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                                  :, :t, :
                                  ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T+1,n_emb)
            x = self.encoder(src=x, mask=self.mask)
            # (B,T+1,n_emb)
            x = x[:, 1:, :]
            # (B,T,n_emb)
        else:
            # encoder
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                if cond_obs_emb.dim() == 2: cond_obs_emb = cond_obs_emb.unsqueeze(1)  # 变成[B, 1, n_emb] czy
                # (B,To,n_emb)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[
                                  :, :tc, :
                                  ]  # each position maps to a (learnable) vector
            cond_embeddings = cond_embeddings.to(self.model_dtype)
            position_embeddings = position_embeddings.to(self.model_dtype)
            # print('cond_embeddings.dtype:', cond_embeddings.dtype)
            # print('position_embeddings.dtype:', position_embeddings.dtype)
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x
            # (B,T_cond,n_emb)

            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                                  :, :t, :
                                  ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask
            )
            # (B,T,n_emb)

        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x

class FlowMatchingActionHead(nn.Module):
    """
    Flow Matching based action head that generates continuous actions via conditional flow matching process.
    
    Based on: https://arxiv.org/abs/2409.01083 "Affordance-based Robot Manipulation with Flow Matching"
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=2,
        action_chunk=5,
        num_flow_steps=5,
        sigma=0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk = action_chunk
        self.num_flow_steps = num_flow_steps
        self.sigma = sigma
        
        print('sigma of FlowMatchingActionHead:', sigma)
        # Flow matching velocity field predictor
        # Note: We need to account for potential timestep embedding concatenation
        # self.velocity_predictor = NoisePredictionModel(
        #     transformer_hidden_dim=hidden_dim*action_dim,  # +hidden_dim for timestep
        #     hidden_dim=hidden_dim, 
        #     action_dim=action_dim
        # )

        self.velocity_predictor = TransformerForDiffusion(
                input_dim=action_dim,
                output_dim=action_dim,
                horizon=action_chunk,
                cond_dim=hidden_dim,
                # n_obs_steps=1,
                # n_emb=hidden_dim
            )
        vp = self.velocity_predictor 
        self.model_device = vp.input_emb.weight.device
        self.model_dtype  = vp.input_emb.weight.dtype

        # Time encoder for flow matching
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)
        
        # Initialize flow matcher
        if ConditionalFlowMatcher is not None:
            self.flow_matcher = ConditionalFlowMatcher(sigma=sigma)
        else:
            self.flow_matcher = None
            print("Warning: Flow matcher not initialized due to missing torchcfm")

    # def sample_flow_trajectory(self, ground_truth_actions):
    #     """
    #     Samples noise and applies flow matching to ground-truth actions to produce flow trajectory.
    #     Returns noise, noisy actions, and the corresponding flow timestep embeddings.
    #     """
    #     # ground_truth_actions: ground-truth actions
    #     # - shape: (batch_size, chunk_len, action_dim)
    #     batch_size = ground_truth_actions.shape[0]
    #     device = ground_truth_actions.device
        
    #     # Sample random noise with shape equal to actions
    #     x0 = torch.randn(size=(batch_size, self.action_chunk, self.action_dim), 
    #                     device=device, dtype=ground_truth_actions.dtype)
        
    #     if self.flow_matcher is not None:
    #         # Use flow matching to sample trajectory points
    #         timestep, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, ground_truth_actions)
    #     else:
    #         # Fallback to simple noise addition if flow matcher not available
    #         timestep = torch.rand(batch_size, device=device)
    #         xt = x0
    #         ut = ground_truth_actions - x0

    #     # Get flow timestep embeddings
    #     flow_timestep_embeddings = self.time_encoder(timestep).to(xt.dtype).to(xt.device)
    #     flow_timestep_embeddings = flow_timestep_embeddings.unsqueeze(1)  # (B, 1, hidden_dim)

    #     return_dict = dict(
    #         noise=ut,  # velocity field
    #         noisy_actions=xt,  # current trajectory point
    #         flow_timestep_embeddings=flow_timestep_embeddings,  # time embeddings
    #         timestep=timestep,
    #     )

    #     return return_dict
    def load_cluster_centers(self, cluster_centers):
        print('loading cluster_centers in FlowMatchingActionHead...')
        self.cluster_centers = cluster_centers

    def select_random_cluster_center(self, batch_size=1):
        """
        随机选择一个聚类中心作为初始化
        
        Returns:
            selected_center: 选择的聚类中心轨迹
        """
        if self.cluster_centers is None:
            return None if batch_size == 1 else [None] * batch_size
        
        
        import random
        
        cluster_ids = list(self.cluster_centers.keys())
        
        if batch_size == 1:
            # 单个选择
            cluster_id = random.choice(cluster_ids)
            selected_center = self.cluster_centers[cluster_id]
            print(f"随机选择聚类 {cluster_id} 作为初始化")
            return [selected_center]
        else:
            # 批量选择
            selected_cluster_ids = random.choices(cluster_ids, k=batch_size)
            selected_centers = [self.cluster_centers[cluster_id] for cluster_id in selected_cluster_ids]
            print(f"随机选择 {batch_size} 个聚类中心作为初始化: {selected_cluster_ids}")
            return selected_centers

    def select_cluster_center_by_prediction(self, logits, batch_size, device):
        # 使用新增的头部预测0-99范围的数字
        predicted_indices = torch.argmax(logits, dim=-1)  # 获取预测的序号
        
        # 根据预测的序号选择聚类中心
        selected_centers = []
        for idx in predicted_indices:
            if idx < len(self.cluster_centers):
                selected_centers.append(self.cluster_centers[idx.item()])
            else:
                # 如果预测的序号超出聚类中心数量，使用随机初始化
                selected_centers.append(torch.randn(self.action_chunk, self.action_dim))
        
        return torch.stack(selected_centers, dim=0).to(device=device, dtype=self.model_dtype)

    def sample_flow_trajectory(self, ground_truth_actions, action_feature=None):
        """
        Samples noise and applies flow matching to ground-truth actions to produce flow trajectory.
        Returns noise, noisy actions, and the corresponding flow timestep embeddings.
        """
        # # ground_truth_actions: ground-truth actions
        # # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        
        if hasattr(self, 'cluster_centers') and self.cluster_centers is not None:
            try:
                if action_feature is not None:
                    x0 = self.select_cluster_center_by_prediction(action_feature, batch_size, device)
                    print(f"✅ 训练时使用预测的聚类中心批量初始化，batch_size={batch_size}")
                else:
                    selected_centers = self.select_random_cluster_center(batch_size)
                    # x0 = torch.tensor(selected_centers, device=device, dtype=ground_truth_actions.dtype)
                    x0 = torch.stack(selected_centers, dim=0).to(device=device, dtype=self.model_dtype)
                    print(f"✅ 训练时使用随机的聚类中心批量初始化，batch_size={batch_size}")
            except Exception as e:
                print(f"❌ 训练时聚类中心初始化失败，使用随机初始化: {e}")
                x0 = torch.randn(size=(batch_size, self.action_chunk, self.action_dim), 
                            device=device, dtype=self.model_dtype)
        else:
            # 原有的随机初始化
            x0 = torch.randn(size=(batch_size, self.action_chunk, self.action_dim), 
                            device=device, dtype=self.model_dtype)
            print("⚠️ 使用随机初始化（无聚类中心）")
        
        if self.flow_matcher is not None:
            # Use flow matching to sample trajectory points
            timestep, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, ground_truth_actions)
        else:
            # Fallback to simple noise addition if flow matcher not available
            timestep = torch.rand(batch_size, device=device)
            xt = x0
            ut = ground_truth_actions - x0                

        # Get flow timestep embeddings
        flow_timestep_embeddings = self.time_encoder(timestep).to(self.model_dtype).to(device)
        flow_timestep_embeddings = flow_timestep_embeddings.unsqueeze(1)  # (B, 1, hidden_dim)

        return_dict = dict(
            noise=ut,  # velocity field
            noisy_actions=xt,  # current trajectory point
            flow_timestep_embeddings=flow_timestep_embeddings,  # time embeddings
            timestep=timestep,
        )

        return return_dict

    def predict_velocity(self, obs_cond, xt, timestep=None):
        """
        Given a batch of last hidden Transformer layer embeddings, predicts the velocity field.
        
        Args:
            obs_cond: last hidden states of Transformer corresponding to action tokens in sequence
            timestep: Optional timestep for time-conditioned prediction
        """


        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # batch_size = obs_cond.shape[0]
        # rearranged_obs_cond = obs_cond.flatten(start_dim=1)
        model_dtype = self.velocity_predictor.input_emb.weight.dtype
        obs_cond = obs_cond.to(model_dtype)
        xt = xt.to(model_dtype)
        if timestep is not None:
            timestep = timestep.to(model_dtype)

        # print('obs_cond:', obs_cond.dtype)
        # print('xt:', xt.dtype)
        # print('model_dtype:', self.model_dtype)

        velocity_pred = self.velocity_predictor(xt, timestep, obs_cond)
            
        return velocity_pred


    def generate_actions(self, obs_conditioning, num_steps=None):
        """
        Generate actions using flow matching sampling.
        
        Args:
            obs_conditioning: Observation conditioning from LLM
            num_steps: Number of sampling steps (default: self.num_flow_steps)
        
        Returns:
            Generated action trajectory
        """
        if num_steps is None:
            num_steps = self.num_flow_steps
            
        batch_size = obs_conditioning.shape[0]
        device = obs_conditioning.device
        
        # Initialize with random noise
        x0 = torch.randn(batch_size, self.action_chunk, self.action_dim, device=device)
        trajectory = x0
        
        # Iterative sampling using flow matching
        for i in range(num_steps):
            timestep = torch.tensor([i / num_steps], device=device).expand(batch_size)
            
            # Get velocity prediction with timestep conditioning
            velocity_pred = self.predict_velocity(obs_conditioning, timestep)
            
            # Update trajectory using velocity field
            if i == 0:
                trajectory = velocity_pred * (1.0 / num_steps) + x0
            else:
                trajectory = velocity_pred * (1.0 / num_steps) + trajectory
                
        return trajectory


class Qwen3VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    config: Qwen3VLConfig
    _checkpoint_conversion_mapping = {}

    # gyn add
    
    def __init__(
        self,
        config,
        action_dim: int = 2,
        action_chunk: int = 5,        # 对应 Qwen2 的 action_chunk
        flow_matching_policy: bool = True,
        num_flow_steps: int = 5,
        action_former: bool = False,
        query_action_layer: int = 4,
        sigma=0,
        ar_lambda_loss: float = 1.0,
        sde_mode='cps',
        **kwargs,
    ):
        print(">>> [DEBUG] using PATCHED Qwen3VLForConditionalGeneration, action_dim =", action_dim)
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Flow Matching 基本参数
        self.action_dim = action_dim
        self.action_chunk = action_chunk
        self.flow_matching_policy = flow_matching_policy
        self.num_flow_steps = num_flow_steps
        self.ar_lambda_loss = ar_lambda_loss
        self.sde_mode=sde_mode

        # 归一化区间 [min, max]
        self.min = [[-1.0] * self.action_dim for _ in range(self.action_chunk)]
        self.max = [[ 1.0] * self.action_dim for _ in range(self.action_chunk)]

        if self.flow_matching_policy:
            hidden_size = config.text_config.hidden_size
            self.flow_matching_model = FlowMatchingActionHead(
                input_dim=hidden_size,
                hidden_dim=hidden_size,
                action_dim=self.action_dim,
                action_chunk=self.action_chunk,
                num_flow_steps=self.num_flow_steps,
                sigma=0.0,
            )
            self.input_wp_encoder = MultiLayerEmbedding(embedding_dim=hidden_size)
            self.action_former = action_former
            self.query_action_layer = query_action_layer

            if self.action_former:
                if self.query_action_layer == 1:
                    self.query_multihead_attn = nn.MultiheadAttention(
                        embed_dim=hidden_size,
                        num_heads=4,
                        batch_first=True,
                    )
                else:
                    self.query_multihead_multi_attn = ActionFormerHead(
                        hidden_dim=hidden_size,
                        num_heads=4,
                        num_blocks=self.query_action_layer,
                    )
                self.query_action = nn.Parameter(torch.empty(1, 1, hidden_size))
                nn.init.normal_(self.query_action, mean=0.0, std=0.02)

        self.rope_deltas = None
        self.post_init()


    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,

        # === gyn Flow Matching 相关新增 ===
        gt_waypoints: Optional[torch.Tensor] = None,
        input_waypoints: Optional[torch.Tensor] = None,
        arrive: Optional[torch.Tensor] = None,
        train: bool = True,
        train_branch: str = "fm",        # "fm" or "ar"
        special_token2id: Optional[Dict[str, int]] = None,
        grpo_mode: Optional[bool] = False, # czy
        next_sampled_actions_list=None,# czy, grpo先验
        sampled_actions_list=None,# czy, grpo先验
        select_noise=None, # 如果使用grpo推理，需要保证noise一致

        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        
        print('train_branch:', train_branch)
        if train_branch:
            assert all(x == train_branch[0] for x in train_branch), f"Mixed train_branch values in batch: {train_branch}"
            train_branch = train_branch[0]  # 如果列表不为空，取第一个元素作为代表
        else:
            train_branch = 'fm'  # 默认值，如果列表为空
        if special_token2id is not None:
            self.special_token2id = special_token2id

        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
            TODO: Add example
        """

        # outputs = self.model(
        #     input_ids=input_ids,
        #     pixel_values=pixel_values,
        #     pixel_values_videos=pixel_values_videos,
        #     image_grid_thw=image_grid_thw,
        #     video_grid_thw=video_grid_thw,
        #     position_ids=position_ids,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     cache_position=cache_position,
        #     **kwargs,
        # )

        # hidden_states = outputs[0]

        # # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(hidden_states[:, slice_indices, :])

        # loss = None
        # if labels is not None:
        #     loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        # return Qwen3VLCausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     rope_deltas=outputs.rope_deltas,
        # )
        if self.flow_matching_policy and train_branch =='fm':
            print('running flow matching branch...')
        # =============================================================
        # =============================================================
            if train is False:
                print("grpo_mode:", grpo_mode)

                # ======================
                # 非 GRPO 推理分支
                # ======================
                if not grpo_mode:
                    # 从 kwargs 取 num_samples，一次采样多少条轨迹
                    num_samples = kwargs.get("num_samples", 1)

                    batch_size = input_ids.shape[0]
                    assert batch_size == 1, "当前实现假定推理时 batch_size=1"

                    device = input_ids.device

                    # 1. 基本配置
                    output_attentions = (
                        output_attentions if output_attentions is not None else self.config.output_attentions
                    )
                    output_hidden_states = (
                        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                    )
                    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

                    # 2. 构建 inputs_embeds = 文本 + 图像 + 视频
                    if inputs_embeds is None:
                        inputs_embeds = self.model.embed_tokens(input_ids)  # (1, L, H)

                        if pixel_values is not None:
                            pixel_values = pixel_values.type(self.visual.dtype)
                            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                            n_image_features = image_embeds.shape[0]
                            if n_image_tokens != n_image_features:
                                raise ValueError(
                                    f"Image tokens {n_image_tokens} != image features {n_image_features}"
                                )
                            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                            image_mask = image_mask.to(inputs_embeds.device)
                            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                        if pixel_values_videos is not None:
                            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                            n_video_features = video_embeds.shape[0]
                            if n_video_tokens != n_video_features:
                                raise ValueError(
                                    f"Video tokens {n_video_tokens} != video features {n_video_features}"
                                )
                            video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                            video_mask = video_mask.to(inputs_embeds.device)
                            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

                        if attention_mask is not None:
                            attention_mask = attention_mask.to(inputs_embeds.device)

                    # 3. 植入 input_waypoints（B=1）
                    if input_waypoints is not None:
                        # (6,D) → (1,6,D)
                        if input_waypoints.dim() == 2:
                            if input_waypoints.shape[0] != 6:
                                raise ValueError(f"input_waypoints dim=2 but first dim !=6: {input_waypoints.shape}")
                            input_waypoints = input_waypoints.unsqueeze(0)
                        if input_waypoints.dim() != 3 or input_waypoints.shape[0] != 1:
                            raise ValueError(f"input_waypoints must be (1,6,D) now, got {input_waypoints.shape}")

                        if "<input_pos1>" in self.special_token2id:
                            start_id = self.special_token2id["<input_pos1>"]
                        else:
                            start_id = 151657

                        for offset in range(6):
                            tok_id = start_id + offset
                            token_mask = (input_ids == tok_id)  # (1,L)
                            if not token_mask.any():
                                continue
                            cur_wp = input_waypoints[:, offset, :].unsqueeze(1)  # (1,1,D_wp)
                            cur_wp = cur_wp.to(self.input_wp_encoder.layer1.weight.dtype)
                            wp_emb = self.input_wp_encoder(cur_wp)              # (1,1,H) 或 (1,H)
                            if wp_emb.dim() == 3:
                                wp_emb = wp_emb.squeeze(1)                      # (1,H)
                            wp_emb = wp_emb.to(inputs_embeds.dtype)
                            inputs_embeds[token_mask] = wp_emb[0]

                    inputs_embeds_dtype = inputs_embeds.dtype
                    inputs_embeds_device = inputs_embeds.device

                    # 4. 计算 position_ids
                    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
                        if (
                            (cache_position is not None and cache_position[0] == 0)
                            or self.rope_deltas is None
                            or (past_key_values is None or past_key_values.get_seq_length() == 0)
                        ):
                            position_ids, rope_deltas = self.get_rope_index(
                                input_ids, image_grid_thw, video_grid_thw, attention_mask
                            )
                            self.rope_deltas = rope_deltas
                        else:
                            _, L, _ = inputs_embeds.shape
                            delta = (
                                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                                if cache_position is not None
                                else 0
                            )
                            position_ids = torch.arange(L, device=inputs_embeds.device).view(1, -1)
                            if cache_position is not None:
                                delta = delta.repeat_interleave(1, dim=0)  # 因为 B=1
                            position_ids = position_ids + delta
                            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

                    # 5. LLM 前向，得到 hidden_states
                    outputs = self.model(
                        input_ids=None,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        cache_position=cache_position,
                    )
                    hidden_states = outputs[0]  # (1,L,H)

                    # 6. 提取 action_feature (1,H)
                    if self.action_former:
                        query_action = self.query_action.expand(1, -1, -1)
                        if self.query_action_layer == 1:
                            action_feature, _ = self.query_multihead_attn(
                                query_action, hidden_states, hidden_states
                            )
                        else:
                            action_feature = self.query_multihead_multi_attn(
                                query_action, hidden_states, hidden_states
                            )
                        action_feature = action_feature.squeeze(1)  # (1,H)
                    else:
                        action_feature = hidden_states[:, -1, :]     # (1,H)

                    action_feature = action_feature.to(inputs_embeds_dtype)

                    # 7. Flow matching：只在这里扩成 num_samples 条条件 + 噪声
                    num_flow_steps = self.flow_matching_model.num_flow_steps
                    print("num_flow_steps:", num_flow_steps)

                    # 条件扩展：(1,H) → (num_samples,H)
                    cond = action_feature.expand(num_samples, -1)   # (num_samples,H)

                    # 噪声初始化：(num_samples,C,D)
                    noise = torch.randn(
                        size=(
                            num_samples,
                            self.flow_matching_model.action_chunk,
                            self.flow_matching_model.action_dim,
                        ),
                        device=inputs_embeds_device,
                        dtype=inputs_embeds_dtype,
                    )
                    curr_flow_trajectory = noise  # (num_samples,C,D)

                    with torch.no_grad():
                        for i in range(num_flow_steps):
                            t = torch.tensor(
                                [i / num_flow_steps], device=inputs_embeds_device, dtype=inputs_embeds_dtype
                            )
                            v = self.flow_matching_model.predict_velocity(
                                cond, curr_flow_trajectory, t
                            )  # (num_samples,C,D)
                            curr_flow_trajectory = curr_flow_trajectory + v * (1.0 / num_flow_steps)

                    # 8. 从 (num_samples,C,D) 恢复多条 waypoint 轨迹
                    pred_waypoints_batch = self.recover_waypoints_from_pred(curr_flow_trajectory)
                    # pred_waypoints_batch: (num_samples, N_wp, D_wp)

                    # arrive_pred 简化成全 0，保持接口
                    arrive_pred = torch.zeros(num_samples, device=device)

                    return pred_waypoints_batch, arrive_pred
           
                else: # grpo mode!!
                    # 推理模式 - 使用flow matching生成action
                    noise = torch.randn(
                        size=(1, self.flow_matching_model.action_chunk, self.flow_matching_model.action_dim), 
                        device=input_ids.device, dtype=torch.bfloat16
                    )
                    
                    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
                    output_hidden_states = (
                        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                    )
                    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
                    
                    if inputs_embeds is None:
                        #额外的token embedding
                        inputs_embeds = self.model.embed_tokens(input_ids)
                        if pixel_values is not None:
                            pixel_values = pixel_values.type(self.visual.dtype)
                            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                            n_image_features = image_embeds.shape[0]
                            if n_image_tokens != n_image_features:
                                raise ValueError(
                                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                                )

                            mask = input_ids == self.config.image_token_id
                            mask_unsqueezed = mask.unsqueeze(-1)
                            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                            image_mask = mask_expanded.to(inputs_embeds.device)

                            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                        if pixel_values_videos is not None:
                            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                            n_video_features = video_embeds.shape[0]
                            if n_video_tokens != n_video_features:
                                raise ValueError(
                                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                                )

                            mask = input_ids == self.config.video_token_id
                            mask_unsqueezed = mask.unsqueeze(-1)
                            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                            video_mask = mask_expanded.to(inputs_embeds.device)

                            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

                        if attention_mask is not None:
                            attention_mask = attention_mask.to(inputs_embeds.device)
                    
                    if input_waypoints is not None:
                        if '<input_pos1>' in self.special_token2id:
                            start_id = self.special_token2id['<input_pos1>']
                        else:
                            start_id=151657
                        for add_id in range(start_id,start_id+6):
                            assert add_id in input_ids
                            cur_input_waypoint = input_waypoints[:,add_id-start_id,:].unsqueeze(1)
                            cur_input_waypoint = cur_input_waypoint.type(self.input_wp_encoder.layer1.weight.dtype)
                            input_waypoint_embedding = self.input_wp_encoder(cur_input_waypoint)
                            input_waypoint_mask = (
                                (input_ids == add_id)
                                .unsqueeze(-1)
                                .expand_as(inputs_embeds)
                                .to(inputs_embeds.device)
                            )
                            input_waypoint_mask = input_ids == add_id
                            input_waypoint_mask_unsqueezed = input_waypoint_mask.unsqueeze(-1)
                            input_waypoint_mask_expanded = input_waypoint_mask_unsqueezed.expand_as(inputs_embeds)
                            input_waypoint_mask_expanded = input_waypoint_mask_expanded.to(inputs_embeds.device)
                            input_waypoint_embedding = input_waypoint_embedding.to(inputs_embeds.device, inputs_embeds.dtype)
                            inputs_embeds = inputs_embeds.masked_scatter(input_waypoint_mask_expanded, input_waypoint_embedding)

                    inputs_embeds_dtype = inputs_embeds.dtype
                    inputs_embeds_device = inputs_embeds.device
                    

                    # 4. 获取position_ids
                    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
                        if (
                            (cache_position is not None and cache_position[0] == 0)
                            or self.rope_deltas is None
                            or (past_key_values is None or past_key_values.get_seq_length() == 0)
                        ):
                            position_ids, rope_deltas = self.get_rope_index(
                                input_ids,
                                image_grid_thw,
                                video_grid_thw,
                                attention_mask,
                            )
                            self.rope_deltas = rope_deltas
                        else:
                            batch_size, seq_length, _ = current_inputs_embeds.shape
                            delta = (
                                (cache_position[0] + self.rope_deltas).to(current_inputs_embeds.device)
                                if cache_position is not None
                                else 0
                            )
                            position_ids = torch.arange(seq_length, device=current_inputs_embeds.device)
                            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                            if cache_position is not None:
                                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                            position_ids = position_ids.add(delta)
                            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                            
                    # 5. 通过LLM获取hidden states
                    outputs = self.model(
                        input_ids=None,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        # past_key_values=past_key_values,
                        past_key_values=None, # czy
                        inputs_embeds=inputs_embeds,
                        # use_cache=use_cache,
                        use_cache=False, # czy
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        cache_position=cache_position,
                        # diffusion_attention_mask=flow_attention_mask
                    )
                    hidden_states = outputs[0]

                    if self.action_former:
                        query_action = self.query_action.expand(hidden_states.shape[0], -1, -1)
                        if self.query_action_layer==1:
                            action_feature, _ = self.query_multihead_attn(query_action, hidden_states, hidden_states)
                        else:
                            action_feature = self.query_multihead_multi_attn(query_action, hidden_states, hidden_states)                
                        action_feature = action_feature.squeeze(1) # torch.Size([2, 1536])
                    else:
                        hidden_states = outputs[0]
                        action_feature = hidden_states[:,-1,:] # 取最后一个
                    batch_size, seq_length, _ = inputs_embeds.shape
                    action_feature = action_feature.to(inputs_embeds_dtype)
                    # Flow matching推理 - 逐步迭代推理
                    # curr_flow_trajectory = noise # 初始化的轨迹位置
                    # 尝试使用聚类中心进行初始化
                    if select_noise is None:
                        if hasattr(self, 'cluster_centers') and self.cluster_centers is not None:
                            try:
                                curr_flow_trajectory = self.select_random_cluster_center()
                                curr_flow_trajectory = torch.stack(curr_flow_trajectory, dim=0).to(device=inputs_embeds_device, dtype=inputs_embeds_dtype)
                                # print("使用聚类中心初始化轨迹")
                            except Exception as e:
                                print(f"聚类中心初始化失败，使用随机初始化: {e}")
                                curr_flow_trajectory = noise
                        else:
                            curr_flow_trajectory = noise # 初始化的轨迹位置
                    else:
                        print(f"使用之前的初始化 select_noise")
                        curr_flow_trajectory = select_noise
                    
                    noise_copy = copy.deepcopy(curr_flow_trajectory)
                    
                    
                    curr_flow_trajectory = curr_flow_trajectory.expand(batch_size, -1, -1)
                    actions_list = [curr_flow_trajectory]
                    all_log_probs = []
                    all_sample_mean = []
                    all_std = []

                    # 逐步迭代推理，从t=0到t=1
                    # 可以通过diffusion_infer_step参数控制步数，减少内存使用
                    num_flow_steps = self.flow_matching_model.num_flow_steps
                    print('num_flow_steps:', num_flow_steps)
                    
                    # 初始化 SDE 调度器
                    # from sde_with_logprob import ConditionalFlowMatcherWithSigmaSchedule
                    if not hasattr(self, 'sde_scheduler'):
                        # self.sde_scheduler = ConditionalFlowMatcherWithSigmaSchedule(num_inference_steps=num_flow_steps,
                        # noise_level=self.noise_level, device=input_ids.device)
                        self.sde_scheduler = ConditionalFlowMatcherWithSigmaSchedule_CPS(num_inference_steps=num_flow_steps,
                        noise_level=self.noise_level, device=input_ids.device)

                    # for i, timestep in enumerate(self.sde_scheduler.timesteps[:-1]): # czy 不包括最后一个
                    # for i, timestep in enumerate(self.sde_scheduler.timesteps[1:]): # czy
                    for i, timestep in enumerate(self.sde_scheduler.timesteps[:-1]): # czy
                        timestep = timestep.to(input_ids.device).unsqueeze(0)
                        # velocity_pred = self.flow_matching_model.predict_velocity(action_feature, curr_flow_trajectory, timestep)
                        velocity_pred = self.flow_matching_model.predict_velocity(action_feature, curr_flow_trajectory, 1-timestep) # debug

                        if self.add_noise_step_num and i >= self.add_noise_step_num:
                            new_traj = curr_flow_trajectory + velocity_pred * (1.0 / num_flow_steps)
                            log_prob = torch.zeros_like(log_prob, device=velocity_pred.device, dtype=velocity_pred.dtype) # ODE步骤的log_prob为0
                            
                            mean = new_traj  # ODE的mean就是预测值
                            std = torch.zeros_like(std, device=velocity_pred.device, dtype=velocity_pred.dtype)  # ODE的std为0
                        else:
                            print(f'only add {self.add_noise_step_num} noise step!!!!')
                            if sampled_actions_list is None:
                                new_traj, log_prob, mean, std = self.sde_scheduler.sde_step_with_logprob(
                                    v_t=velocity_pred,
                                    timestep=timestep, # TODO: check
                                    x_t=curr_flow_trajectory,
                                )
                            else:
                                new_traj, log_prob, mean, std = self.sde_scheduler.sde_step_with_logprob(
                                    v_t=velocity_pred,
                                    timestep=timestep,
                                    x_t=sampled_actions_list[i],  # x_t=curr_flow_trajectory,
                                    x_t_new=next_sampled_actions_list[i] # 如果sampled_actions_list不为nan，则传入
                                )              
                        # curr_flow_trajectory = curr_flow_trajectory + velocity_pred * (1.0 / num_flow_steps)  # before
                        curr_flow_trajectory = new_traj # grpo

                        actions_list.append(new_traj)
                        all_log_probs.append(log_prob)
                        all_sample_mean.append(mean)
                        all_std.append(std)

                        curr_flow_trajectory = curr_flow_trajectory.to(inputs_embeds_dtype)
                    
                    del action_feature, outputs, attention_mask
                    # 13. 恢复最终路径点
                    pred_waypoints = self.recover_waypoints_from_pred(curr_flow_trajectory)
                    return pred_waypoints, self.sde_scheduler.timesteps, actions_list, all_log_probs, all_sample_mean, all_std, noise_copy
            else:
                # 训练模式
                delta_waypoints = torch.zeros_like(gt_waypoints)
                delta_waypoints[:, 0] = gt_waypoints[:, 0]
                delta_waypoints[:, 1:] = gt_waypoints[:, 1:] - gt_waypoints[:, :-1]
                min_vals = torch.tensor(self.min, dtype=delta_waypoints.dtype, device=delta_waypoints.device).unsqueeze(0) # [1,5,4]
                max_vals = torch.tensor(self.max, dtype=delta_waypoints.dtype, device=delta_waypoints.device).unsqueeze(0) # [1,5,4]
                # 标准化
                delta_waypoints = (delta_waypoints - min_vals) / (max_vals - min_vals + 1e-8)
                delta_waypoints = delta_waypoints * 2 - 1 # [-1,1] 
                delta_waypoints = delta_waypoints.to(self.visual.dtype)
                
                output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
                output_hidden_states = (
                    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                )
                return_dict = return_dict if return_dict is not None else self.config.use_return_dict
                
                #获取语言embeeding 视觉embeeding 时间embeeding concat
                if inputs_embeds is None:
                    #额外的token embedding
                    inputs_embeds = self.model.embed_tokens(input_ids)
                    if pixel_values is not None:
                        pixel_values = pixel_values.type(self.visual.dtype)
                        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                        n_image_features = image_embeds.shape[0]
                        if n_image_tokens != n_image_features:
                            raise ValueError(
                                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                            )

                        mask = input_ids == self.config.image_token_id
                        mask_unsqueezed = mask.unsqueeze(-1)
                        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                        image_mask = mask_expanded.to(inputs_embeds.device)

                        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)


                    if pixel_values_videos is not None:
                        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                        n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                        n_video_features = video_embeds.shape[0]
                        if n_video_tokens != n_video_features:
                            raise ValueError(
                                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                            )

                        mask = input_ids == self.config.video_token_id
                        mask_unsqueezed = mask.unsqueeze(-1)
                        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                        video_mask = mask_expanded.to(inputs_embeds.device)

                        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

                    if attention_mask is not None:
                        attention_mask = attention_mask.to(inputs_embeds.device)
                
                if len(input_waypoints) and input_waypoints is not None:
                    if '<input_pos1>' in self.special_token2id:
                        start_id = self.special_token2id['<input_pos1>']
                    else:
                        start_id=151657
                    for add_id in range(start_id,start_id+6):
                        assert add_id in input_ids
                        cur_input_waypoint = input_waypoints[:,add_id-start_id,:].unsqueeze(1)
                        cur_input_waypoint = cur_input_waypoint.type(self.input_wp_encoder.layer1.weight.dtype)
                        input_waypoint_embedding = self.input_wp_encoder(cur_input_waypoint)
                        input_waypoint_mask = (
                            (input_ids == add_id)
                            .unsqueeze(-1)
                            .expand_as(inputs_embeds)
                            .to(inputs_embeds.device)
                        )
                        input_waypoint_mask = input_ids == add_id
                        input_waypoint_mask_unsqueezed = input_waypoint_mask.unsqueeze(-1)
                        input_waypoint_mask_expanded = input_waypoint_mask_unsqueezed.expand_as(inputs_embeds)
                        input_waypoint_mask_expanded = input_waypoint_mask_expanded.to(inputs_embeds.device)
                        input_waypoint_embedding = input_waypoint_embedding.to(inputs_embeds.device, inputs_embeds.dtype)
                        inputs_embeds = inputs_embeds.masked_scatter(input_waypoint_mask_expanded, input_waypoint_embedding)
                
                # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
                if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
                    # calculate RoPE index once per generation in the pre-fill stage only
                    if (
                        (cache_position is not None and cache_position[0] == 0)
                        or self.rope_deltas is None
                        or (past_key_values is None or past_key_values.get_seq_length() == 0)
                    ):
                        position_ids, rope_deltas = self.get_rope_index(
                            input_ids,
                            image_grid_thw,
                            video_grid_thw,
                            attention_mask,
                        )
                        self.rope_deltas = rope_deltas
                    # then use the prev pre-calculated rope-deltas to get the correct position ids
                    else:
                        batch_size, seq_length, _ = inputs_embeds.shape
                        delta = (
                            (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                            if cache_position is not None
                            else 0
                        )
                        position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                        position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                        if cache_position is not None:  # otherwise `deltas` is an int `0`
                            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                        position_ids = position_ids.add(delta)
                        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

                outputs = self.model(
                    input_ids=None,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                    # diffusion_attention_mask=flow_attention_mask
                )

                inputs_embeds_dtype = inputs_embeds.dtype
                # inputs_embeds_device = inputs_embeds.device

                hidden_states = outputs[0]
                if self.action_former:
                    query_action = self.query_action.expand(hidden_states.shape[0], -1, -1)
                    if self.query_action_layer==1:
                        action_feature, _ = self.query_multihead_attn(query_action, hidden_states, hidden_states)
                    else:
                        action_feature = self.query_multihead_multi_attn(query_action, hidden_states, hidden_states)                
                    action_feature = action_feature.squeeze(1) # torch.Size([2, 1536])
                else:
                    hidden_states = outputs[0]
                    action_feature = hidden_states[:,-1,:] # 取最后一个
                action_feature = action_feature.to(inputs_embeds_dtype)

                # 使用flow matching采样轨迹
                if self.use_cluster_selector_head:
                    cluster_logits = self.cluster_selector_head(action_feature)
                    flow_dict = self.flow_matching_model.sample_flow_trajectory(delta_waypoints.to(self.visual.dtype), cluster_logits)
                else:
                    flow_dict = self.flow_matching_model.sample_flow_trajectory(delta_waypoints.to(self.visual.dtype))
                noise, noisy_actions, timestep, flow_timestep_embeddings = (
                    flow_dict["noise"],
                    flow_dict["noisy_actions"], 
                    flow_dict["timestep"],
                    flow_dict["flow_timestep_embeddings"],
                )
                
                noisy_actions = noisy_actions.to(inputs_embeds_dtype)
                timestep = timestep.to(inputs_embeds_dtype)

                # self.flow_matching_model.velocity_predictor._initialize_weights()
                velocity_pred = self.flow_matching_model.predict_velocity(action_feature, noisy_actions, timestep)
                # Get flow matching velocity prediction MSE loss
                velocity_pred = velocity_pred.reshape(noise.shape)
                loss = nn.functional.mse_loss(velocity_pred, noise.to(inputs_embeds_dtype), reduction="mean")
                print('❗️flow loss',loss)

                # logits = self.lm_head(hidden_states)# czy debug
                return Qwen2VLCausalLMOutputWithPast(
                        loss=loss,
                        logits=None,
                        # logits=logits,# czy debug
                        past_key_values=outputs.past_key_values,
                        hidden_states=outputs.hidden_states,
                        attentions=outputs.attentions,
                        rope_deltas=self.rope_deltas,
                    )

        print('running auto regressive branch...')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
            if pixel_values is None and pixel_values_videos is None:
                # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
                dummy_pixel = torch.zeros(784, 1176).to(self.visual.get_device())
                dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.get_device())
                
                dummy_pixel = dummy_pixel.type(self.visual.get_dtype())
                image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
                # Operates as maksed_scatter for the image tokens
                # However the values are all zeros so it dosen't affect the embeddings.
                # This could avoid deepspeed error when some batch only has texts.
                inputs_embeds += image_embeds.mean() * 0

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        loss = None
        logits = None

        if self.training and (labels is not None):
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten tokens
            shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
            shift_labels = shift_labels.view(-1)

            lce = LigerFusedLinearCrossEntropyLoss()
            loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
        else:
            logits = self.lm_head(hidden_states)
            if labels is not None:
                # Upcast to float if we need to compute the loss to avoid potential precision issues
                logits = logits.float()
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        loss = self.ar_lambda_loss * loss # czy add; 用于balance loss的大小
        print('❗️ar loss',loss)
        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen3VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs


class Qwen3VLVideosProcessorKwargs(VideosKwargs, total=False):
    pass


class Qwen3VLImagesKwargs(Qwen2VLImagesKwargs):
    pass


class Qwen3VLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Qwen3VLImagesKwargs
    videos_kwargs: Qwen3VLVideosProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class Qwen3VLProcessor(Qwen2VLProcessor):
    r"""
    Constructs a Qwen3VL processor which wraps a Qwen3VL image processor and a Qwen2 tokenizer into a single processor.
    [`Qwen3VLProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`] and [`Qwen2TokenizerFast`]. See the
    [`~Qwen3VLProcessor.__call__`] and [`~Qwen3VLProcessor.decode`] for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`Qwen3VLVideoProcessor`], *optional*):
            The video processor is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, video_processor, chat_template, **kwargs)
        self.vision_start_token = (
            "<|vision_start|>" if not hasattr(tokenizer, "vision_start_token") else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<|vision_end|>" if not hasattr(tokenizer, "vision_end_token") else tokenizer.vision_end_token
        )
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen3VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            # If user has not requested video metadata, pop it
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        logger.warning_once(
                            "Qwen3VL requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                            "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                            "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                        )
                        metadata.fps = 24 if metadata.fps is None else metadata.fps

                    # if timestamps are not provided, calculate them
                    curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        self.video_processor.merge_size,
                    )

                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            self.vision_start_token + "<|placeholder|>" * frame_seqlen + self.vision_end_token
                        )
                    if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
                        text[i] = text[i].replace(
                            f"{self.vision_start_token}{self.video_token}{self.vision_end_token}", video_placeholder, 1
                        )
                    else:
                        # vllm may input video token directly
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1

                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    def _calculate_timestamps(self, indices: Union[list[int], np.ndarray], video_fps: float, merge_size: int = 2):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]
        # @JJJYmmm frames are merged by self.merge_size, \
        # so we need to average the timestamps between the first/last frame within the temporal patch
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps


__all__ = [
    "Qwen3VLConfig",
    "Qwen3VLTextConfig",
    "Qwen3VLVisionModel",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLPreTrainedModel",
    "Qwen3VLProcessor",
    "Qwen3VLTextModel",
]
