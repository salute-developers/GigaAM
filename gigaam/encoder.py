import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

try:
    from flash_attn import flash_attn_func

    IMPORT_FLASH = True
except Exception as err:
    IMPORT_FLASH = False
    IMPORT_FLASH_ERR = err

from .streaming import (
    ConformerLayerStreamState,
    EncoderStreamState,
    LayerStreamState,
    ZapformerLayerStreamState,
)
from .utils import apply_masked_flash_attn, apply_rotary_pos_emb


def _stream_tail(x, lengths, width):
    indices = lengths[:, None] + torch.arange(width, device=x.device)
    return x.gather(-1, indices[:, None].expand(-1, x.shape[1], -1))


def _streaming_convolution(x, conv, cache, chunk_size, position, lengths=None):
    """Convolve with cached history and future context bounded by each block."""
    radius = conv.kernel_size[0] // 2
    past = cache.convolution
    if past is None:
        past = x.new_zeros(x.shape[0], x.shape[1], radius)
    joined = torch.cat((past, x), dim=-1)
    cache.convolution = joined[..., -radius:] if radius else joined[..., :0]
    if lengths is not None:
        cache.convolution = _stream_tail(joined, lengths, radius)
    weight, bias = conv.weight, conv.bias
    y = F.conv1d(joined, weight[..., : radius + 1], bias, groups=x.shape[1])
    padded = torch.cat((x, torch.zeros_like(past)), dim=-1)
    position = torch.as_tensor(position, device=x.device).reshape(-1, 1, 1)
    phase = (position + torch.arange(x.shape[-1], device=x.device)) % chunk_size
    for j in range(1, radius + 1):
        visible = (phase + j < chunk_size).to(x.dtype)
        y = y + padded[..., j : j + x.shape[-1]] * weight[:, :, radius + j] * visible
    return y


def _conformer_layer_fwd(
    layer: nn.Module,
    x: Tensor,
    pos_emb: Union[Tensor, List[Tensor]],
    att_mask: Optional[Tensor],
    pad_mask: Optional[Tensor],
) -> Tensor:
    return layer(x=x, pos_emb=pos_emb, att_mask=att_mask, pad_mask=pad_mask)


class StridingSubsampling(nn.Module):
    """
    Strided Subsampling layer used to reduce the sequence length.
    """

    def __init__(
        self,
        subsampling: str,
        kernel_size: int,
        subsampling_factor: int,
        feat_in: int,
        feat_out: int,
        conv_channels: int,
    ):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.subsampling_type = subsampling
        assert self.subsampling_type in ["conv1d", "conv2d"]
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self._stride = 2
        self._kernel_size = kernel_size
        self._padding = (self._kernel_size - 1) // 2

        layers: List[nn.Module] = []
        in_channels = 1 if self.subsampling_type == "conv2d" else feat_in
        subs_conv_class = (
            torch.nn.Conv2d if self.subsampling_type == "conv2d" else torch.nn.Conv1d
        )
        for _ in range(self._sampling_num):
            layers.append(
                subs_conv_class(
                    in_channels=in_channels,
                    out_channels=conv_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._padding,
                )
            )
            layers.append(nn.ReLU())
            in_channels = conv_channels

        out_length = self.calc_output_length(torch.tensor(feat_in))
        if self.subsampling_type == "conv2d":
            self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv = torch.nn.Sequential(*layers)

    def calc_output_length(
        self, lengths: Tensor, num_stages: Optional[int] = None
    ) -> Tensor:
        """
        Valid length after applying ``num_stages`` strided subsampling conv
        stages (defaults to all of them, i.e. the full subsampling output).
        """
        if num_stages is None:
            num_stages = self._sampling_num
        add_pad = 2 * self._padding - self._kernel_size
        lengths = lengths.to(torch.float)
        for _ in range(num_stages):
            lengths = torch.floor((lengths + add_pad) / self._stride + 1.0)
        return lengths.to(dtype=torch.int)

    def _mask_time(self, x: Tensor, lengths: Tensor) -> Tensor:
        """
        Zero out the padded tail along the time axis (dim 2). The subsampling
        convolutions are strided and have a receptive field wider than the
        stride, so the padded frames of shorter samples leak into the last
        valid frames. Left unmasked, the padding is the log-mel floor
        (``log(1e-9) ~= -20.7``) of zero-padded audio, not zero, so a batched
        short sample sees a different boundary than the same sample run alone
        (where conv zero-padding applies instead). Re-zeroing after every conv
        stage keeps the valid frames of batched inference aligned with the
        batch-size-1 result.
        """
        time = torch.arange(x.size(2), device=x.device)
        pad = time[None, :] >= lengths[:, None]
        pad = pad[:, None]
        if x.dim() == 4:
            pad = pad[..., None]
        return x.masked_fill(pad, 0.0)

    def forward_streaming(self, x, lengths, state):
        x = self._mask_time(x.transpose(1, 2), lengths)
        caches = []
        convs = [conv for conv in self.conv if isinstance(conv, nn.Conv1d)]
        for i, conv in enumerate(convs):
            radius = conv.padding[0]
            past = (
                state.frontend[i]
                if state.frontend
                else x.new_zeros(x.shape[0], conv.in_channels, radius)
            )
            joined = torch.cat((past, x), 2)
            caches.append(joined[:, :, -radius:] if radius else joined[:, :, :0])
            x = F.relu(F.conv1d(joined, conv.weight, conv.bias, stride=conv.stride))
            lengths = self.calc_output_length(lengths, 1).to(lengths.dtype)
            x = (
                x
                * (torch.arange(x.shape[2], device=x.device)[None] < lengths[:, None])[
                    :, None
                ]
            )
        state.frontend = caches
        return x.transpose(1, 2), lengths

    def forward(self, x: Tensor, lengths: Tensor, state=None) -> Tuple[Tensor, Tensor]:
        if state is not None:
            return self.forward_streaming(x, lengths, state)
        if self.subsampling_type == "conv2d":
            x = x.unsqueeze(1)
        else:
            x = x.transpose(1, 2)

        cur_len = lengths
        x = self._mask_time(x, cur_len)
        for module in self.conv:
            x = module(x)
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                cur_len = self.calc_output_length(cur_len, 1)
                x = self._mask_time(x, cur_len)

        if self.subsampling_type == "conv2d":
            b, _, t, _ = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        else:
            x = x.transpose(1, 2)
        return x, self.calc_output_length(lengths)


class MultiHeadAttention(nn.Module, ABC):
    """
    Base class of Multi-Head Attention Mechanisms.
    """

    def __init__(
        self, n_head: int, n_feat: int, flash_attn=False, torch_sdpa_attn=False
    ):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.flash_attn = flash_attn
        self.torch_sdpa_attn = torch_sdpa_attn
        if self.flash_attn and not IMPORT_FLASH:
            raise RuntimeError(
                f"flash_attn_func was imported with err {IMPORT_FLASH_ERR}. "
                "Please install flash_attn or use --no_flash flag. "
                "If you have already done this, "
                "--force-reinstall flag might be useful"
            )

    def forward_qkv(
        self, query: Tensor, key: Tensor, value: Tensor, flash: Optional[bool] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Projects the inputs into queries, keys, and values for multi-head attention.
        """
        b = query.size(0)
        q = self.linear_q(query).view(b, -1, self.h, self.d_k)
        k = self.linear_k(key).view(b, -1, self.h, self.d_k)
        v = self.linear_v(value).view(b, -1, self.h, self.d_k)
        flash = self.flash_attn if flash is None else flash
        if flash:
            return q, k, v
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward_attention(
        self, value: Tensor, scores: Tensor, mask: Optional[Tensor]
    ) -> Tensor:
        """
        Computes the scaled dot-product attention given the projected values and scores.
        """
        b = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -10000.0)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(attn, value)
        x = x.transpose(1, 2).reshape(b, -1, self.h * self.d_k)
        return self.linear_out(x)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """
    Relative Position Multi-Head Attention module.
    """

    def __init__(self, n_head: int, n_feat: int):
        super().__init__(n_head, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))

    def rel_shift(self, x: Tensor) -> Tensor:
        b, h, qlen, pos_len = x.size()
        x = torch.nn.functional.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        return x[:, :, 1:].view(b, h, qlen, pos_len)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)
        pos_emb = pos_emb.to(dtype=self.linear_pos.weight.dtype)
        p = self.linear_pos(pos_emb)
        p = p.view(pos_emb.shape[0], -1, self.h, self.d_k).transpose(1, 2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RotaryPositionMultiHeadAttention(MultiHeadAttention):
    """
    Rotary Position Multi-Head Attention module.
    """

    def __init__(self, *args, apply_rope_before_projections: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_rope_before_projections = apply_rope_before_projections

    def rotary_qkv(self, query, key, value, pos_emb, flash=False):
        b, t, _ = value.size()
        query = query.transpose(0, 1).view(t, b, self.h, self.d_k)
        key = key.transpose(0, 1).view(t, b, self.h, self.d_k)
        value = value.transpose(0, 1).view(t, b, self.h, self.d_k)

        cos, sin = pos_emb
        if self.apply_rope_before_projections:
            query, key = apply_rotary_pos_emb(query, key, cos, sin, offset=0)

        q, k, v = self.forward_qkv(
            query.view(t, b, self.h * self.d_k).transpose(0, 1),
            key.view(t, b, self.h * self.d_k).transpose(0, 1),
            value.view(t, b, self.h * self.d_k).transpose(0, 1),
            flash=flash,
        )

        if not self.apply_rope_before_projections:
            order = (1, 0, 2, 3) if flash else (2, 0, 1, 3)
            restore = (1, 0, 2, 3) if flash else (1, 2, 0, 3)
            q, k = apply_rotary_pos_emb(q.permute(*order), k.permute(*order), cos, sin)
            q, k = q.permute(*restore), k.permute(*restore)

        return q, k, v

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: List[Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        b, t, _ = value.size()
        q, k, v = self.rotary_qkv(query, key, value, pos_emb, self.flash_attn)

        if self.flash_attn:
            if mask is None:
                scores = flash_attn_func(q, k, v)
            else:
                scores = apply_masked_flash_attn(q, k, v, mask, self.h, self.d_k)
            scores = scores.view(b, -1, self.h * self.d_k)
            return self.linear_out(scores)
        elif self.torch_sdpa_attn:
            attn_mask = None
            if mask is not None:
                attn_mask = ~mask.unsqueeze(1)

                attn_mask = attn_mask | (~attn_mask.any(dim=-1, keepdim=True))
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            attn_output = attn_output.transpose(1, 2).reshape(b, t, self.h * self.d_k)
            return self.linear_out(attn_output)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1) / math.sqrt(self.d_k))
            return self.forward_attention(v, scores, mask)


class PositionalEncoding(nn.Module, ABC):
    """
    Base class of Positional Encodings.
    """

    def __init__(self, dim: int, base: int):
        super().__init__()
        self.dim = dim
        self.base = base

    @abstractmethod
    def create_pe(self, length: int, device: torch.device) -> Optional[Tensor]:
        pass

    def extend_pe(self, length: int, device: torch.device):
        """
        Extends the positional encoding buffer to process longer sequences.
        """
        pe = self.create_pe(length, device)
        if pe is None:
            return
        if hasattr(self, "pe"):
            self.pe = pe
        else:
            self.register_buffer("pe", pe, persistent=False)


class RelPositionalEmbedding(PositionalEncoding):
    """
    Relative Positional Embedding module.
    """

    def create_pe(self, length: int, device: torch.device) -> Optional[Tensor]:
        """
        Creates the relative positional encoding matrix.
        """
        if hasattr(self, "pe") and self.pe.shape[1] >= 2 * length - 1:
            return None
        positions = torch.arange(length - 1, -length, -1, device=device).unsqueeze(1)
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.dim, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=pe.device)
            * -(math.log(10000.0) / self.dim)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor]:
        input_len = x.size(1)
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        return x, self.pe[:, start_pos:end_pos]


class RotaryPositionalEmbedding(PositionalEncoding):
    """
    Rotary Positional Embedding module.
    """

    def create_pe(self, length: int, device: torch.device) -> Optional[Tensor]:
        """
        Creates or extends the rotary positional encoding matrix.
        """
        if hasattr(self, "pe") and self.pe.size(0) >= 2 * length:
            return None
        positions = torch.arange(0, length, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        t = torch.arange(length, device=positions.device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(positions.device)
        return torch.cat([emb.cos()[:, None, None, :], emb.sin()[:, None, None, :]])

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, List[Tensor]]:
        cos_emb = self.pe[0 : x.shape[1]]
        half_pe = self.pe.shape[0] // 2
        sin_emb = self.pe[half_pe : half_pe + x.shape[1]]
        return x, [cos_emb, sin_emb]


class ConformerConvolution(nn.Module):
    """
    Conformer Convolution module.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        norm_type: str,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        assert norm_type in ["batch_norm", "layer_norm"]
        self.norm_type = norm_type
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
            bias=True,
        )
        self.batch_norm = (
            nn.BatchNorm1d(d_model)
            if norm_type == "batch_norm"
            else nn.LayerNorm(d_model)
        )
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(x)
        if self.norm_type == "batch_norm":
            x = self.batch_norm(x)
        else:
            x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class ConformerFeedForward(nn.Module):
    """
    Conformer Feed Forward module.
    """

    def __init__(self, d_model: int, d_ff: int, use_bias=True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.activation(self.linear1(x)))


class ConformerLayer(nn.Module):
    """
    Conformer Layer module.
    This module combines several submodules including feed forward networks,
    depthwise separable convolution, and multi-head self-attention
    to form a single Conformer block.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        self_attention_model: str,
        n_heads: int = 16,
        conv_norm_type: str = "batch_norm",
        conv_kernel_size: int = 31,
        flash_attn: bool = False,
        apply_rope_before_projections: bool = True,
    ):
        super().__init__()
        self.fc_factor = 0.5
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
        )
        self.norm_self_att = nn.LayerNorm(d_model)
        if self_attention_model == "rotary":
            self.self_attn: nn.Module = RotaryPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                flash_attn=flash_attn,
                torch_sdpa_attn=not flash_attn,
                apply_rope_before_projections=apply_rope_before_projections,
            )
        else:
            assert not flash_attn, "Not supported flash_attn for rel_pos"
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
            )
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        pos_emb: Union[Tensor, List[Tensor]],
        att_mask: Optional[Tensor] = None,
        pad_mask: Optional[Tensor] = None,
    ) -> Tensor:
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + x * self.fc_factor

        x = self.norm_self_att(residual)
        x = self.self_attn(x, x, x, pos_emb, mask=att_mask)
        residual = residual + x

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask)
        residual = residual + x

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + x * self.fc_factor

        x = self.norm_out(residual)
        return x


class StreamingEncoder(nn.Module, ABC):
    """Shared native packet execution and encoder export hooks."""

    feat_in: int
    feat_out: int

    def __init__(self):
        super().__init__()
        self.export_language_dim = 0
        self.export_head = False
        self.export_emotion = False
        self._onnx_context = None
        self._onnx_caches: List[Tensor] = []

    @abstractmethod
    def _create_layer_states(self) -> List[LayerStreamState]:
        """Create the architecture's typed layer caches."""

    @property
    def subsampling_factor(self):
        return self.frontend.subsampling_factor

    def _forward_batch(self, signal, lengths, chunk, left, lang_embedding=None):
        state = self.create_stream_state(chunk, left)
        multiple = self.subsampling_factor
        x = F.pad(signal, (0, (-signal.shape[-1]) % multiple))
        x, lengths = self.frontend(x.transpose(1, 2), lengths, state)
        x = self._prepare_export_features(x)
        encoded = self._forward_chunk(x, lengths, state, lang_embedding)
        original = (
            signal.shape[-1] + self.subsampling_factor - 1
        ) // self.subsampling_factor
        return encoded[:, :original].transpose(1, 2), lengths

    def create_stream_state(self, chunk, left):
        return EncoderStreamState(
            chunk_size=chunk, left_context=left, layers=self._create_layer_states()
        )

    @contextmanager
    def onnx_export_mode(self):
        saved = [
            (module, name, getattr(module, name))
            for module in self.modules()
            for name in ("flash_attn", "torch_sdpa_attn")
            if hasattr(module, name)
        ]
        try:
            for module, name, _ in saved:
                setattr(module, name, False)
            yield
        finally:
            for module, name, value in saved:
                setattr(module, name, value)

    def input_example(self, batch_size: int = 8, seqlen: int = 200):
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        cached = isinstance(self._onnx_context, tuple)
        if cached:
            chunk, left, dynamic = self._onnx_context
            values = [
                torch.zeros(1, chunk, self.feat_out, device=device, dtype=dtype),
                torch.tensor([chunk], device=device),
            ]
            if dynamic:
                values.append(torch.tensor([chunk], device=device))
            values += [
                torch.zeros(1, dtype=torch.long, device=device) for _ in range(2)
            ]
        else:
            features = torch.randn(
                batch_size, self.feat_in, seqlen, device=device, dtype=dtype
            )
            lengths = torch.randint(1, seqlen + 1, (batch_size,), device=device)
            lengths[0] = seqlen
            values = [features, lengths]
            if isinstance(self._onnx_context, dict):
                values.append(
                    torch.tensor([self._onnx_context["chunk"]], device=device)
                )
        if self.export_language_dim:
            language = values[0].new_zeros(
                values[0].shape[0], self.export_language_dim, dtype=torch.float32
            )
            language[:, 0] = 1
            values.append(language)
        return tuple(values + self._onnx_caches if cached else values)

    def input_names(self) -> List[str]:
        cached = isinstance(self._onnx_context, tuple)
        names = (
            ["features", "feature_lengths"]
            if cached or self.export_head
            else ["audio_signal", "length"]
        )
        if cached:
            if self._onnx_context[2]:
                names.append("context_frames")
            names += ["position", "cache_valid"]
        if isinstance(self._onnx_context, dict):
            names.append("context_frames")
        if self.export_language_dim:
            names.append("language_vector")
        if cached:
            names += [f"cache_{i}" for i in range(len(self._onnx_caches))]
        return names

    def output_names(self) -> List[str]:
        if isinstance(self._onnx_context, tuple):
            return [
                "encoded",
                "encoded_lengths",
                "next_position",
                "next_cache_valid",
                *[f"next_cache_{i}" for i in range(len(self._onnx_caches))],
            ]
        if self.export_head:
            return (
                ["probs"] if self.export_emotion else ["log_probs", "encoded_lengths"]
            )
        return ["encoded", "encoded_len"]

    def dynamic_axes(self) -> Optional[Dict[str, Dict[int, str]]]:
        if isinstance(self._onnx_context, tuple):
            return {"features": {1: "frames"}, "encoded": {2: "frames"}}
        inputs, outputs = self.input_names(), self.output_names()
        axes = {
            inputs[0]: {0: "batch_size", 2: "seq_len"},
            inputs[1]: {0: "batch_size"},
            outputs[0]: {
                0: "batch_size",
                1 if self.export_head else 2: "seq_len",
            },
        }
        if len(outputs) > 1:
            axes[outputs[1]] = {0: "batch_size"}
        if self.export_language_dim:
            axes["language_vector"] = {0: "batch_size"}
        return axes

    def _prepare_export_features(self, features):
        return features

    def forward_for_export(self, features, lengths, *values, lang_embedding=None):
        """Run the encoder with exported cache tensors."""
        cached = isinstance(self._onnx_context, tuple)
        original = features.shape[1]
        if cached:
            chunk, left, dynamic = self._onnx_context
            if dynamic:
                context, values = values[0], values[1:]
                features = self._prepare_export_features(features)
                chunk = torch.where(
                    context[0] > 0, context[0], torch._shape_as_tensor(features)[1]
                ).clamp_min(1)
            position, cache_valid, *values = values

        if isinstance(self._onnx_context, dict):
            context = values[0]
            factor = self.subsampling_factor
            frames = (torch._shape_as_tensor(features)[2] + factor - 1) // factor
            chunk = torch.where(context[0] > 0, context[0], frames).clamp_min(1)
            stack_multiple = max(getattr(self, "downsampling_factor", (1,)))
            chunk = ((chunk + stack_multiple - 1) // stack_multiple) * stack_multiple
            return self._forward_batch(
                features,
                lengths,
                chunk,
                self._onnx_context["left_context"],
                lang_embedding,
            )

        if cached:
            state = self.create_stream_state(chunk, left)
            state.position, state.valid = (
                (position, cache_valid)
                if getattr(self, "_triton_batch", False)
                else (position[0], cache_valid[0])
            )
            state.restore(values)
            encoded = self._forward_chunk(
                features, lengths, state, lang_embedding
            ).transpose(1, 2)[..., :original]
            outputs = (
                lengths,
                state.position + lengths,
                (state.valid + lengths).clamp_max(left),
                *state.tensors(),
            )
        else:
            kwargs = (
                {"chunk_size": int(self._onnx_context / self.frame_shift_ms)}
                if self._onnx_context
                else {}
            )
            if lang_embedding is not None:
                kwargs["lang_embedding"] = lang_embedding
            encoded, *outputs = type(self).forward(self, features, lengths, **kwargs)
        return (encoded, *outputs)

    def forward_frontend_for_export(self, features, lengths, *front):
        state = EncoderStreamState(frontend=list(front))
        x, lengths = self.frontend(features.transpose(1, 2), lengths, state)
        return (x, lengths, *state.frontend)

    @torch.inference_mode()
    def _stream_forward(self, signal, lengths, state, final, lang_embedding=None):
        size = self.subsampling_factor * state.chunk_size
        x, count = state.take_ready(signal, size, final, axis=2)
        if not count:
            out = signal.new_empty(1, self.feat_out, 0)
            return out, lengths.new_zeros(1), state
        x = F.pad(x, (0, (-count) % size))
        x, encoded_len = self.frontend(
            x.transpose(1, 2), lengths.new_tensor([count]), state
        )
        x = self._forward_chunk(x, encoded_len, state, lang_embedding)[
            :, : int(encoded_len[0])
        ]
        state.position += x.shape[1]
        return x.transpose(1, 2), encoded_len, state


class ConformerEncoder(StreamingEncoder):
    """
    Conformer Encoder module.
    This module encapsulates the entire Conformer encoder architecture,
    consisting of a StridingSubsampling layer, positional embeddings, and
    a stack of Conformer Layers.
    It serves as the main component responsible for processing speech features.
    """

    def __init__(
        self,
        feat_in: int = 64,
        n_layers: int = 16,
        d_model: int = 768,
        subsampling: str = "conv2d",
        subs_kernel_size: int = 3,
        subsampling_factor: int = 4,
        ff_expansion_factor: int = 4,
        self_attention_model: str = "rotary",
        n_heads: int = 16,
        pos_emb_max_len: int = 5000,
        conv_norm_type: str = "batch_norm",
        conv_kernel_size: int = 31,
        flash_attn: bool = False,
        activation_checkpointing: bool = False,
        apply_rope_before_projections: bool = True,
        rope_base: Optional[int] = None,
        use_causal_conv_chunking: bool = False,
    ):
        super().__init__()
        self.feat_in = feat_in
        self.feat_out = d_model
        self.activation_checkpointing = activation_checkpointing
        self.use_causal_conv_chunking = use_causal_conv_chunking
        assert self_attention_model in [
            "rotary",
            "rel_pos",
        ], f"Not supported attn = {self_attention_model}"

        self.pre_encode = StridingSubsampling(
            subsampling=subsampling,
            kernel_size=subs_kernel_size,
            subsampling_factor=subsampling_factor,
            feat_in=feat_in,
            feat_out=d_model,
            conv_channels=d_model,
        )

        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rotary":
            self.pos_enc: PositionalEncoding = RotaryPositionalEmbedding(
                d_model // n_heads, pos_emb_max_len if rope_base is None else rope_base
            )
        else:
            self.pos_enc = RelPositionalEmbedding(d_model, pos_emb_max_len)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_model * ff_expansion_factor,
                self_attention_model=self_attention_model,
                n_heads=n_heads,
                conv_norm_type=conv_norm_type,
                conv_kernel_size=conv_kernel_size,
                flash_attn=flash_attn,
                apply_rope_before_projections=apply_rope_before_projections,
            )
            self.layers.append(layer)

    def forward(
        self,
        audio_signal: Tensor,
        length: Tensor,
        state: Optional[EncoderStreamState] = None,
        final: bool = False,
        chunk_size=None,
        left_context=32,
    ):
        if state is not None:
            return self._stream_forward(audio_signal, length, state, final)
        if chunk_size is not None and chunk_size > 0:
            return self._forward_batch(audio_signal, length, chunk_size, left_context)
        if not hasattr(self.pos_enc, "pe"):
            self.pos_enc.extend_pe(self.pos_emb_max_len, audio_signal.device)

        audio_signal, length = self.pre_encode(
            x=audio_signal.transpose(1, 2), lengths=length
        )

        max_len = audio_signal.size(1)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)

        pad_mask = torch.arange(0, max_len, device=audio_signal.device).expand(
            length.size(0), -1
        ) < length.unsqueeze(-1)

        att_mask = None
        if audio_signal.shape[0] > 1:
            att_mask = pad_mask.unsqueeze(1).repeat([1, max_len, 1])
            att_mask = torch.logical_and(att_mask, att_mask.transpose(1, 2))
            att_mask = ~att_mask

        pad_mask = ~pad_mask

        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                audio_signal = checkpoint(
                    _conformer_layer_fwd,
                    layer,
                    audio_signal,
                    pos_emb,
                    att_mask,
                    pad_mask,
                    use_reentrant=False,
                )
            else:
                audio_signal = layer(
                    x=audio_signal,
                    pos_emb=pos_emb,
                    att_mask=att_mask,
                    pad_mask=pad_mask,
                )

        return audio_signal.transpose(1, 2), length

    def _create_layer_states(self) -> List[LayerStreamState]:
        return [ConformerLayerStreamState() for _ in self.layers]

    @property
    def frontend(self):
        return self.pre_encode

    def _stream_layer(self, layer, x, state, stream, valid):
        position, chunk_size = stream.position, stream.chunk_size
        left_context, cache_valid = stream.left_context, stream.valid
        residual = x + 0.5 * layer.feed_forward1(layer.norm_feed_forward1(x))
        query = layer.norm_self_att(residual)
        # Preserve normalized attention history.
        cached = state.attention
        n_past = 0 if cached is None else cached.shape[1]
        keys = query if cached is None else torch.cat((cached, query), dim=1)
        attn = layer.self_attn
        t = keys.shape[1]
        cos, sin = (
            self.pos_enc.pe[: self.pos_enc.pe.shape[0] // 2],
            self.pos_enc.pe[self.pos_enc.pe.shape[0] // 2 :],
        )
        # Rotate keys at global positions.
        start = position - n_past
        if cache_valid is None:
            cos, sin = cos[start : start + t], sin[start : start + t]
        else:
            positions = (torch.arange(t, device=x.device) + start).clamp_min(0).float()
            frequencies = self.pos_enc.base ** (
                -torch.arange(0, self.pos_enc.dim, 2, device=x.device).float()
                / self.pos_enc.dim
            )
            angles = positions[:, None] * frequencies[None]
            angles = torch.cat((angles, angles), -1)[:, None, None]
            cos, sin = angles.cos(), angles.sin()
        q, k, v = attn.rotary_qkv(keys, keys, keys, (cos, sin))
        # Build chunk and history masks.
        qp = (
            torch.arange(position, position + x.shape[1], device=x.device)
            if cache_valid is None
            else position + torch.arange(x.shape[1], device=x.device)
        )[:, None]
        kp = (
            torch.arange(start, start + t, device=x.device)
            if cache_valid is None
            else start + torch.arange(t, device=x.device)
        )[None]
        same_chunk = qp // chunk_size == kp // chunk_size
        history = (kp >= qp - left_context) & (kp <= qp)
        nonempty = keys.abs().amax(dim=-1) >= 1e-5
        allowed = same_chunk[None] | (history[None] & nonempty[:, None])
        past_valid = torch.arange(n_past, device=x.device) >= (
            0 if cache_valid is None else n_past - cache_valid
        )
        past_valid = past_valid[None] & nonempty[:, :n_past]
        key_valid = torch.cat((past_valid, valid), dim=1)
        allowed = allowed & key_valid[:, None]
        # Give padded queries a self-attention entry to keep ONNX softmax finite.
        allowed = allowed | (~valid[:, :, None] & (qp == kp))
        attended = F.scaled_dot_product_attention(
            q[:, :, n_past:], k, v, attn_mask=allowed[:, None]
        )
        residual = residual + attn.linear_out(
            attended.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        )
        # Save committed attention history.
        state.attention = keys[:, -left_context:] if left_context else keys[:, :0]
        # Convolve within the active chunk.
        conv = layer.conv
        glu = F.glu(
            conv.pointwise_conv1(layer.norm_conv(residual).transpose(1, 2)), dim=1
        )
        if valid is not None:
            glu = glu * valid[:, None]
        y = _streaming_convolution(
            glu, conv.depthwise_conv, state, chunk_size, position
        )
        # Apply convolution and residual normalization.
        y = (
            conv.batch_norm(y)
            if conv.norm_type == "batch_norm"
            else conv.batch_norm(y.transpose(1, 2)).transpose(1, 2)
        )
        residual = residual + conv.pointwise_conv2(conv.activation(y)).transpose(1, 2)
        return layer.norm_out(
            residual + 0.5 * layer.feed_forward2(layer.norm_feed_forward2(residual))
        )

    def _forward_chunk(self, x, lengths, state, lang_embedding=None):
        if state.valid is None:
            needed = state.position + x.shape[1]
            if (
                not hasattr(self.pos_enc, "pe")
                or self.pos_enc.pe.shape[0] // 2 < needed
            ):
                self.pos_enc.extend_pe(max(self.pos_emb_max_len, 2 * needed), x.device)
        valid = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        for layer, cache in zip(self.layers, state.layers):
            x = self._stream_layer(layer, x, cache, state, valid)
        return x


# Adapted from Xiaomi Corp., Daniel Povey / Zengwei Yao, Apache-2.0.
# https://www.apache.org/licenses/LICENSE-2.0


class SwishL(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return (
            F.silu(0.639521598815918 * x - 2.087040662765503)
            - 0.021007252857089043 * x
            + 0.2303316444158554
        )


class SwishR(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return (
            F.silu(0.683202862739563 * x - 0.31602001190185547)
            - 0.012898006476461887 * x
            + 0.133248575988922
        )


def _zap_activation(name, variant="r"):
    if name.lower() == "silu":
        return nn.SiLU()
    return SwishL() if variant == "l" else SwishR()


class RMSNormTorch(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = (x * x).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RmsNorm(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.eps = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        x_sq = torch.mean(x * x, dim=-1, keepdim=True) + self.eps * self.eps
        return x * (self.scale / x_sq.sqrt())


class OrthogonalLinear(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, weight_rms: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.normal_(self.weight, std=weight_rms)
        self.weight_scale = in_channels ** (-0.5) / weight_rms

    def forward(self, x: Tensor, transpose: bool = False) -> Tensor:
        weight = self.weight * self.weight_scale
        if transpose:
            weight = weight.t()
        return F.linear(x, weight, None)


class ActivationAndLinear(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, activation="swish"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        self.activation = _zap_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(self.activation(x), self.weight, self.bias)


class RoPETable(nn.Module):

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.cos = nn.Parameter(freqs.cos()[None, None, :, :], requires_grad=False)
        self.sin = nn.Parameter(freqs.sin()[None, None, :, :], requires_grad=False)
        self.max_seq_len = max_seq_len

    @staticmethod
    def key(dim: int, max_seq_len: int, base: float = 10000.0) -> str:
        return f"ed{dim}_S{max_seq_len}_b{int(base)}"

    def forward(self, positions: Tensor) -> Tuple[Tensor, Tensor]:
        cos = self.cos.index_select(2, positions)
        sin = self.sin.index_select(2, positions)
        return (cos, sin)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    B, H, S, D = x.shape
    x_pair = x.reshape(B, H, S, D // 2, 2)
    x0 = x_pair[..., 0]
    x1 = x_pair[..., 1]
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    return torch.stack((y0, y1), dim=-1).reshape(B, H, S, D)


def _zap_tuple(value, count=None):
    values = (
        tuple(map(int, value.split(",")))
        if isinstance(value, str)
        else ((value,) if isinstance(value, int) else tuple(value))
    )
    if count is not None:
        if len(values) == 1:
            values *= count
    return values


def _zapformer_conv2d(x, layer, padding=None, native=False):
    if native or not torch.onnx.is_in_onnx_export():
        if padding is None:
            return layer(x)
        return F.conv2d(
            x, layer.weight, layer.bias, layer.stride, padding, groups=layer.groups
        )
    padding = layer.padding if padding is None else padding
    if layer.kernel_size == (1, 1):
        y = F.linear(x.permute(0, 2, 3, 1), layer.weight[:, :, 0, 0], layer.bias)
        return y.permute(0, 3, 1, 2)

    batch, channels, time, freq = x.shape
    kt, kf = layer.kernel_size
    st, sf = layer.stride
    pt, pf = padding
    time = (time + 2 * pt - kt) // st + 1
    freq = (freq + 2 * pf - kf) // sf + 1
    patches = F.unfold(x, layer.kernel_size, padding=padding, stride=layer.stride)
    if layer.groups == 1:
        y = F.linear(patches.transpose(1, 2), layer.weight.flatten(1), layer.bias)
        y = y.transpose(1, 2)
    else:
        weights = layer.weight.reshape(1, channels, kt * kf, 1)
        y = (patches.reshape(batch, channels, kt * kf, -1) * weights).sum(2)
        y = y + layer.bias[None, :, None]
    return y.reshape(batch, layer.out_channels, time, freq)


class ZapformerConvNeXt(nn.Module):
    def __init__(self, channels, causal, activation="swish"):
        super().__init__()
        self.causal = causal
        kernel = 3 if causal else 7
        self.depthwise_conv = nn.Conv2d(
            channels, channels, kernel, groups=channels, padding=0 if causal else 3
        )
        self.pointwise_conv1 = nn.Conv2d(channels, channels * 3, 1)
        self.activation = _zap_activation(activation, "l")
        self.pointwise_conv2 = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x):
        residual = x
        if self.causal:
            x = F.pad(x, (1, 1, 2, 0))
        x = _zapformer_conv2d(x, self.depthwise_conv)
        x = self.activation(_zapformer_conv2d(x, self.pointwise_conv1))
        return residual + _zapformer_conv2d(x, self.pointwise_conv2)


class ZapformerFrontend(nn.Module):
    def __init__(self, feat_in, dim, channels, causal, activation="swish"):
        super().__init__()
        self.causal = causal
        c1, c2, c3 = channels
        self.conv1 = nn.Conv2d(1, c1, 3, padding=0 if causal else 1)
        self.conv2 = nn.Conv2d(c1, c2, 3, stride=2, padding=(1, 0))
        self.conv3 = nn.Conv2d(c2, c3, 3, stride=2, padding=(1, 0))
        self.subsampling_factor = self.conv2.stride[0] * self.conv3.stride[0]
        self.activation = _zap_activation(activation)
        self.convnext = ZapformerConvNeXt(c3, causal, activation)
        self.out = nn.Linear(c3 * (((feat_in - 1) // 2 - 1) // 2), dim)

    @staticmethod
    def _tail(joined, lengths, width):
        index = lengths[:, None] + torch.arange(width, device=joined.device)
        return joined.gather(
            2, index[:, None, :, None].expand(-1, joined.shape[1], -1, joined.shape[-1])
        )

    def forward(self, x, lengths, state=None):
        native = getattr(self, "_export_native_conv", False)
        valid = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        x = (x * valid.unsqueeze(-1)).unsqueeze(1)
        caches = []
        for i, conv in enumerate((self.conv1, self.conv2, self.conv3)):
            if state is None:
                x = F.pad(x, (1, 1, 2, 0)) if i == 0 and self.causal else x
                x = _zapformer_conv2d(x, conv)
            else:
                context = 2 if i == 0 else 1
                past = (
                    state.frontend[i]
                    if state.frontend
                    else x.new_zeros(x.shape[0], x.shape[1], context, x.shape[3])
                )
                joined = torch.cat((past, x), 2)
                caches.append(self._tail(joined, lengths, context))
                if native:
                    x = _zapformer_conv2d(joined, conv, (0, int(i == 0)), native=True)
                else:
                    x = _zapformer_conv2d(
                        F.pad(joined, (1, 1, 0, 0)) if i == 0 else joined,
                        conv,
                        padding=(0, 0),
                    )
            x = self.activation(x)
            if i:
                lengths = (lengths + 1) // 2
            valid = torch.arange(x.shape[2], device=x.device)[None] < lengths[:, None]
            x = x * valid[:, None, :, None]
        if state is None:
            x = self.convnext(x)
        else:
            past = (
                state.frontend[3]
                if state.frontend
                else x.new_zeros(x.shape[0], x.shape[1], 2, x.shape[3])
            )
            joined = torch.cat((past, x), 2)
            caches.append(self._tail(joined, lengths, 2))
            conv = self.convnext
            y = (
                _zapformer_conv2d(joined, conv.depthwise_conv, (0, 1), native=True)
                if native
                else _zapformer_conv2d(F.pad(joined, (1, 1, 0, 0)), conv.depthwise_conv)
            )
            y = conv.activation(
                _zapformer_conv2d(y, conv.pointwise_conv1, native=native)
            )
            x = x + _zapformer_conv2d(y, conv.pointwise_conv2, native=native)
            state.frontend = caches
        return self.out(x.transpose(1, 2).flatten(2)) * valid.unsqueeze(-1), lengths


class ZapformerAttention(nn.Module):
    def __init__(self, dim, heads, qdim, vdim, dropout):
        super().__init__()
        self.heads, self.qdim, self.vdim = heads, qdim, vdim
        self.in_norm = RMSNormTorch(dim)
        self.qk_in_proj = nn.Linear(dim, 2 * heads * qdim)
        self.vg_in_proj = nn.Linear(dim, 3 * heads * vdim)
        self.out_proj = nn.Linear(heads * vdim, dim)
        self.dropout = dropout
        self.torch_sdpa_attn = True

    def forward(self, x_qk, x_vg, cos, sin, mask, offset=0):
        b, t, _ = x_qk.shape
        q, k = self.qk_in_proj(self.in_norm(x_qk)).chunk(2, -1)
        q, k = [
            apply_rope(
                z.reshape(b, t, self.heads, self.qdim).transpose(1, 2),
                cos.to(z.dtype),
                sin.to(z.dtype),
            )
            for z in (q, k)
        ]
        v, gate_in, gate_out = self.vg_in_proj(x_vg).chunk(3, -1)
        v = (v * gate_in.sigmoid()).reshape(b, t, self.heads, self.vdim).transpose(1, 2)
        if self.torch_sdpa_attn:
            # Native fully masked rows have uniform weights, including padded queries.
            all_masked = mask.all(-1, keepdim=True)[:, None]
            out = F.scaled_dot_product_attention(
                q[:, :, offset:].masked_fill(all_masked, 0),
                k,
                v,
                attn_mask=~mask[:, None] | all_masked,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            scores = (q[:, :, offset:] * (self.qdim**-0.5)) @ k.transpose(-1, -2)
            weights = F.softmax(scores.masked_fill(mask[:, None], -10000.0), dim=-1)
            weights = F.dropout(weights, self.dropout, self.training)
            out = weights @ v
        out = out.transpose(1, 2).reshape(b, t - offset, self.heads * self.vdim)
        return self.out_proj(out * gate_out[:, offset:].sigmoid())


class ZapformerFeedForward(nn.Module):
    def __init__(self, dim, multiple, activation="swish"):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim * multiple)
        self.activation = _zap_activation(activation)
        self.activation_dropout = nn.Dropout(0.05)
        self.out_proj = nn.Linear(dim * multiple, dim)

    def forward(self, x):
        return self.out_proj(self.activation_dropout(self.activation(self.in_proj(x))))


class ZapformerConvolution(nn.Module):
    def __init__(self, dim, kernel, chunk, activation="swish"):
        super().__init__()
        self.chunk, self.context = chunk, (kernel - 1) // 2
        self.in_proj = nn.Linear(dim, 2 * dim)
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel, groups=dim, padding=0 if chunk > 0 else self.context
        )
        self.weighted_mean = nn.Module()
        self.weighted_mean.weights = nn.Parameter(torch.zeros(dim))
        self.out_proj = ActivationAndLinear(dim, dim, activation)

    def forward(
        self,
        x,
        valid,
        context,
        cache=None,
        cache_len=0,
        chunk_size=None,
        position=0,
        batched=False,
    ):
        chunk_size = self.chunk if chunk_size is None else chunk_size
        x, gate = self.in_proj(x).chunk(2, -1)
        x = x * valid.unsqueeze(-1)
        history = (
            x if cache is None or cache.mean is None else torch.cat((cache.mean, x), 1)
        )
        accum = (
            history.double()
            if cache is None and x.dtype == torch.float64
            else history.float()
        )
        mean = (context.to(accum.dtype) @ accum) / context.sum(
            -1, keepdim=True
        ).clamp_min(1)
        mean = mean * self.weighted_mean.weights
        y = x.transpose(1, 2)
        if cache is not None:
            cache.mean = history[:, -cache_len:] if cache_len else history[:, :0]
            lengths = valid.sum(1) if batched else None
            if batched:
                cache.mean = _stream_tail(
                    history.transpose(1, 2), lengths, cache_len
                ).transpose(1, 2)
            y = _streaming_convolution(
                y, self.depthwise_conv, cache, chunk_size, position, lengths
            )
        elif chunk_size > 0:
            b, c, t = y.shape
            y = F.pad(y, (0, (-t) % chunk_size)).reshape(b, c, -1, chunk_size)
            left = y[..., -self.context :].roll(1, dims=2)
            left = torch.cat((torch.zeros_like(left[:, :, :1]), left[:, :, 1:]), dim=2)
            y = F.pad(torch.cat((left, y), dim=-1), (0, self.context)).transpose(1, 2)
            y = F.conv1d(
                y.reshape(-1, c, chunk_size + 2 * self.context),
                self.depthwise_conv.weight,
                self.depthwise_conv.bias,
                groups=c,
            )
            y = (
                y.reshape(b, -1, c, chunk_size)
                .transpose(1, 2)
                .reshape(b, c, -1)[..., :t]
            )
        else:
            y = F.conv1d(
                y,
                self.depthwise_conv.weight,
                self.depthwise_conv.bias,
                padding=self.context,
                groups=y.shape[1],
            )
        return self.out_proj((y.transpose(1, 2) + mean) * gate.sigmoid())


class ZapformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        qdim,
        vdim,
        multiple,
        kernel,
        chunk,
        dropout,
        activation="swish",
    ):
        super().__init__()
        self.feed_forward1 = ZapformerFeedForward(dim, multiple, activation)
        self.feed_forward2 = (
            ZapformerFeedForward(dim, multiple, activation) if kernel > 0 else None
        )
        self.self_attn = ZapformerAttention(dim, heads, qdim, vdim, 0.1)
        self.conv_module = (
            ZapformerConvolution(dim, kernel, chunk, activation) if kernel > 0 else None
        )
        self.norm = RMSNormTorch(dim)
        self.dropout = dropout

    def forward(self, x, cos, sin, valid, allowed, chunk_size=None):
        residual = x
        qk = x
        x = x + F.dropout(self.feed_forward1(x), self.dropout, self.training)
        mask = ~allowed[None] | ~valid[:, None]
        x = x + F.dropout(
            self.self_attn(qk, x, cos, sin, mask), self.dropout, self.training
        )
        if self.conv_module is not None:
            x = x + F.dropout(
                self.conv_module(x, valid, ~mask, chunk_size=chunk_size),
                self.dropout,
                self.training,
            )
        if self.feed_forward2 is not None:
            x = x + F.dropout(self.feed_forward2(x), self.dropout, self.training)
        return self.norm(residual + 0.25 * (x - residual))


class ZapformerStack(nn.Module):
    def __init__(
        self,
        input_dim,
        dim,
        layers,
        heads,
        qdim,
        vdim,
        multiple,
        kernel,
        chunk,
        dropout,
        rope_bank,
        activation="swish",
    ):
        super().__init__()
        self.proj = OrthogonalLinear(input_dim, dim)
        self.layers = nn.ModuleList(
            [
                ZapformerLayer(
                    dim, heads, qdim, vdim, multiple, kernel, chunk, dropout, activation
                )
                for _ in range(layers)
            ]
        )
        self.residual_scales = nn.Parameter(
            torch.tensor([-1.0] + [1.0 / layers] * layers)
        )
        self.input_scale = nn.Parameter(torch.ones(1))
        self.rope_bank = rope_bank

    def forward(self, x, cos, sin, valid, allowed, layer_hook=None, chunk_size=None):
        residual = x
        x = self.proj(x)
        bypass = self.residual_scales[0] * x
        x = self.input_scale * x
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, valid, allowed, chunk_size)
            if layer_hook is not None:
                x = layer_hook(x, i)
            bypass = bypass + self.residual_scales[i + 1] * x
        return residual + self.proj(bypass, transpose=True)


class ZapformerEncoder(StreamingEncoder):
    """Checkpoint-compatible RoPE Zapformer for offline whole-utterance inference.

    Causal checkpoints retain their chunk/window attention and convolution rules.
    All operations support autograd; native training regularizers are not included.
    """

    def __init__(
        self,
        in_channels=64,
        frontend_channels="8,32,128",
        downsampling_factor="1,2,4,2",
        embed_dim=None,
        encoder_dim="320,512,768,512",
        num_encoder_layers="6,8,14,8",
        query_head_dim=None,
        value_head_dim="96,96,96,96",
        num_heads="4,4,4,4",
        feedforward_multiple="4,4,3,4",
        conv_params="31,31,15,31",
        causal=False,
        causal_frontend=False,
        chunk_size=-1,
        window_size=(-1, -1),
        chunk_window=(-1, -1),
        max_seq_len=10000,
        dropout=0.1,
        flash_attn=False,
        dual_mode=False,
        activation="swish",
    ):
        super().__init__()
        if flash_attn:
            raise NotImplementedError(
                "flash_attn is not supported for Zapformer; use PyTorch SDPA instead."
            )
        self.dual_mode = dual_mode
        self.language_film = None
        if dual_mode:
            chunks = _zap_tuple(chunk_size)
            windows = (
                [tuple(w) for w in window_size]
                if not isinstance(window_size[0], int)
                else [tuple(window_size)] * len(chunks)
            )
            self._context_modes = dict(zip(chunks, windows))
            causal, chunk_size, window_size = False, -1, (-1, -1)
        self.feat_in = in_channels
        self.causal, self.causal_frontend = causal, causal_frontend
        self.downsampling_factor = ds = _zap_tuple(downsampling_factor)
        n = len(ds)
        dims, layers, heads, vdims, multiples, kernels = [
            _zap_tuple(v, n)
            for v in (
                encoder_dim,
                num_encoder_layers,
                num_heads,
                value_head_dim,
                feedforward_multiple,
                conv_params,
            )
        ]
        self.layer_dims = [
            dim for dim, count in zip(dims, layers) for _ in range(count)
        ]
        qdims = (
            _zap_tuple(query_head_dim, n)
            if query_head_dim is not None
            else tuple(d // h for d, h in zip(dims, heads))
        )
        self.chunk_size = chunk_size if causal else -1
        self.window_size, self.chunk_window = _zap_tuple(window_size), _zap_tuple(
            chunk_window
        )
        dim = embed_dim or dims[0]
        self.feat_out = dim
        self.encoder_embed = ZapformerFrontend(
            in_channels,
            dim,
            _zap_tuple(frontend_channels, 3),
            causal_frontend,
            activation,
        )
        self.encoder = nn.Module()
        self.encoder.rope_bank = nn.Module()
        self.encoder.rope_bank.tables = nn.ModuleDict(
            {
                RoPETable.key(q, max_seq_len): RoPETable(q, max_seq_len)
                for q in set(qdims)
            }
        )
        self.encoder.encoders = nn.ModuleList(
            [
                ZapformerStack(
                    d * dim,
                    e,
                    l,
                    h,
                    q,
                    v,
                    m,
                    k,
                    self.chunk_size // d if causal else -1,
                    dropout,
                    self.encoder.rope_bank,
                    activation,
                )
                for d, e, l, h, q, v, m, k in zip(
                    ds, dims, layers, heads, qdims, vdims, multiples, kernels
                )
            ]
        )
        self.encoder.out_norm = RmsNorm()
        self.qdims, self.max_seq_len = qdims, max_seq_len

    def _allowed(self, size, device, chunk, window_size):
        q = torch.arange(size, device=device)[:, None]
        k = q.T
        if chunk <= 0:
            return torch.ones(size, size, device=device, dtype=torch.bool)
        return self._attention_mask(q, k, chunk, window_size)

    def _attention_mask(self, q, k, chunk, window_size=None):
        qc, kc = q // chunk, k // chunk
        base_allowed = qc == kc
        allowed = base_allowed
        if window_size is None and self.dual_mode and isinstance(chunk, Tensor):
            for mode, window in self._context_modes.items():
                if mode > 0:
                    allowed = allowed | (
                        (chunk == mode) & self._attention_mask(q, k, chunk, window)
                    )
            return allowed
        for left, right, query, key in (
            (*(self.window_size if window_size is None else window_size), q, k),
            (*self.chunk_window, qc, kc),
        ):
            if (left, right) != (-1, -1):
                allowed = allowed | (
                    (key >= query - left) if left >= 0 else torch.ones_like(allowed)
                ) & ((key <= query + right) if right >= 0 else torch.ones_like(allowed))
        return allowed

    def forward(
        self,
        audio_signal,
        length,
        state: Optional[EncoderStreamState] = None,
        final: bool = False,
        chunk_size=None,
        window_size=None,
        lang_embedding=None,
        left_context=32,
    ):
        chunk = state.chunk_size if state is not None else chunk_size
        if chunk is None:
            chunk = -1 if self.dual_mode else self.chunk_size
        window = self._context_modes[chunk] if self.dual_mode else self.window_size
        if state is not None:
            return self._stream_forward(
                audio_signal, length, state, final, lang_embedding
            )
        if chunk_size is not None and chunk_size > 0:
            return self._forward_batch(
                audio_signal, length, chunk_size, left_context, lang_embedding
            )
        x, length = self.encoder_embed(audio_signal.transpose(1, 2), length)
        original = x.shape[1]
        max_ds = max(self.downsampling_factor)
        x = F.pad(x, (0, 0, 0, (-original) % max_ds))
        valid = torch.arange(x.shape[1], device=x.device)[None] < length[:, None]
        allowed = self._allowed(x.shape[1], x.device, chunk, window)
        layer_offset = 0
        for ds, qdim, stack in zip(
            self.downsampling_factor, self.qdims, self.encoder.encoders
        ):
            if ds > 1:
                x = x * valid.unsqueeze(-1)
            x = x.reshape(x.shape[0], -1, x.shape[-1] * ds)
            table = self.encoder.rope_bank.tables[RoPETable.key(qdim, self.max_seq_len)]
            cos, sin = table(torch.arange(x.shape[1], device=x.device))
            hook = (
                (
                    lambda value, index: self.language_film(
                        value, layer_offset + index, lang_embedding
                    )
                )
                if self.language_film is not None
                else None
            )
            x = stack(
                x,
                cos,
                sin,
                valid[:, ::ds],
                allowed[::ds, ::ds],
                hook,
                chunk // ds if chunk > 0 else chunk,
            )
            layer_offset += len(stack.layers)
            x = x.reshape(x.shape[0], -1, x.shape[-1] // ds)
        return self.encoder.out_norm(x[:, :original]).transpose(1, 2), length

    def _create_layer_states(self) -> List[LayerStreamState]:
        return [
            ZapformerLayerStreamState(has_convolution=layer.conv_module is not None)
            for stack in self.encoder.encoders
            for layer in stack.layers
        ]

    def _stream_geometry(
        self,
        attn,
        x,
        valid,
        position,
        cache_valid,
        chunk_size,
        ds,
        n_past,
        history_frames,
    ):
        # Track downsampled attention positions.
        qpos = position + torch.arange(x.shape[1], device=x.device)[None, :, None] * ds
        kpos = (
            position
            - n_past * ds
            + torch.arange(history_frames, device=x.device)[None] * ds
        )
        window = (
            None
            if self.dual_mode and isinstance(chunk_size, Tensor)
            else (
                self._context_modes[chunk_size] if self.dual_mode else self.window_size
            )
        )
        allowed = self._attention_mask(qpos, kpos, chunk_size, window)
        # Mask padding and unfilled history.
        past_valid = (
            torch.ones(1, n_past, device=x.device, dtype=torch.bool)
            if cache_valid is None
            else (
                torch.arange(n_past, device=x.device)[None]
                >= n_past
                - torch.as_tensor(cache_valid, device=x.device).reshape(-1, 1) // ds
            )
        )
        key_valid = torch.cat((past_valid.expand(x.shape[0], -1), valid), dim=1)
        context = allowed & key_valid[:, None]
        # Apply position-aware cached attention.
        table = self.encoder.rope_bank.tables[
            RoPETable.key(attn.qdim, self.max_seq_len)
        ]
        positions = torch.arange(history_frames, device=x.device)[None]
        if cache_valid is not None:
            positions = (
                positions
                - n_past
                + torch.as_tensor(cache_valid, device=x.device).reshape(-1, 1) // ds
            ).clamp_min(0)
        cos, sin = (value[0, 0][positions][:, None] for value in (table.cos, table.sin))
        return context, cos, sin

    def _stream_layer(self, layer, x, cache, state, ds, valid, geometry=None):
        chunk_size, position = state.chunk_size, state.position
        cache_len, cache_valid = state.left_context // ds, state.valid
        batched = isinstance(position, Tensor) and position.ndim == 1
        position = torch.as_tensor(position, device=x.device).reshape(-1, 1, 1)
        original, qk = x, x
        x = x + layer.feed_forward1(x)
        history_q = qk if cache.query is None else torch.cat((cache.query, qk), dim=1)
        history_v = x if cache.value is None else torch.cat((cache.value, x), dim=1)
        n_past = history_q.shape[1] - x.shape[1]
        attn = layer.self_attn
        key = (ds, n_past, attn.qdim)
        if geometry is None or key not in geometry:
            masks = self._stream_geometry(
                attn,
                x,
                valid,
                position,
                cache_valid,
                chunk_size,
                ds,
                n_past,
                history_q.shape[1],
            )
            if geometry is not None:
                geometry[key] = masks
        else:
            masks = geometry[key]
        context, cos, sin = masks
        x = x + attn(history_q, history_v, cos, sin, ~context, offset=n_past)
        # Retain attention query/value caches.
        cache.query = history_q[:, -cache_len:] if cache_len else history_q[:, :0]
        cache.value = history_v[:, -cache_len:] if cache_len else history_v[:, :0]
        if batched:
            lengths = valid.sum(1)
            cache.query, cache.value = (
                _stream_tail(history.transpose(1, 2), lengths, cache_len).transpose(
                    1, 2
                )
                for history in (history_q, history_v)
            )
        # Reuse cached convolution statistics.
        if layer.conv_module is not None:
            x = x + layer.conv_module(
                x,
                valid,
                context,
                cache,
                cache_len,
                chunk_size // ds,
                position.reshape(-1) // ds,
                batched,
            )
        if layer.feed_forward2 is not None:
            x = x + layer.feed_forward2(x)
        return layer.norm(original + 0.25 * (x - original))

    def _prepare_export_features(self, features):
        multiple = max(self.downsampling_factor)
        padding = features.new_zeros(
            features.shape[0], (-features.shape[1]) % multiple, features.shape[2]
        )
        return torch.cat((features, padding), dim=1)

    def _forward_chunk(self, x, encoded_len, state, lang_embedding=None):
        valid_full = (
            torch.arange(x.shape[1], device=x.device)[None] < encoded_len[:, None]
        )
        geometry = {} if x.is_cuda and not torch.onnx.is_in_onnx_export() else None
        idx = 0
        for ds, stack in zip(self.downsampling_factor, self.encoder.encoders):
            if ds > 1:
                x = x * valid_full.unsqueeze(-1)
            x = x.reshape(x.shape[0], -1, x.shape[-1] * ds)
            original = x
            x = stack.proj(x)
            bypass = stack.residual_scales[0] * x
            x = stack.input_scale * x
            for i, layer in enumerate(stack.layers):
                x = self._stream_layer(
                    layer,
                    x,
                    state.layers[idx],
                    state,
                    ds,
                    valid_full[:, ::ds],
                    geometry,
                )
                if self.language_film is not None:
                    x = self.language_film(x, idx, lang_embedding)
                bypass = bypass + stack.residual_scales[i + 1] * x
                idx += 1
            x = original + stack.proj(bypass, transpose=True)
            x = x.reshape(x.shape[0], -1, x.shape[-1] // ds)
        return self.encoder.out_norm(x)

    @property
    def frontend(self):
        return self.encoder_embed
