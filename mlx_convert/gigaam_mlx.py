"""
GigaAM v3 CTC model in MLX.

Architecture:
  - FeatureExtractor: log-mel spectrogram (computed via numpy/MLX at runtime)
  - ConformerEncoder: conv1d subsampling + rotary positional embeddings + 16 conformer layers
  - CTCHead: Conv1d(768, num_classes, kernel=1)
  - CTC greedy decoding

Supports pseudo-streaming via sliding window (like whisper-stream / mlx-audio Parakeet).
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ─────────────────────────── Config ───────────────────────────

@dataclass
class GigaAMConfig:
    model_type: str = "gigaam"
    model_name: str = "v3_ctc"
    sample_rate: int = 16000
    # preprocessor
    features: int = 64
    win_length: int = 320
    hop_length: int = 160
    n_fft: int = 320
    center: bool = False
    # encoder
    feat_in: int = 64
    n_layers: int = 16
    d_model: int = 768
    subsampling: str = "conv1d"
    subs_kernel_size: int = 5
    subsampling_factor: int = 4
    ff_expansion_factor: int = 4
    self_attention_model: str = "rotary"
    pos_emb_max_len: int = 5000
    n_heads: int = 16
    conv_kernel_size: int = 5
    conv_norm_type: str = "layer_norm"
    # head
    head_type: str = "ctc"
    num_classes: int = 34
    # vocabulary
    vocabulary: Optional[List[str]] = None

    @classmethod
    def from_file(cls, path: str) -> "GigaAMConfig":
        with open(path) as f:
            d = json.load(f)
        enc = d.get("encoder", {})
        pre = d.get("preprocessor", {})
        head = d.get("head", {})
        return cls(
            model_type=d.get("model_type", "gigaam"),
            model_name=d.get("model_name", "v3_ctc"),
            sample_rate=d.get("sample_rate", 16000),
            features=pre.get("features", 64),
            win_length=pre.get("win_length", 320),
            hop_length=pre.get("hop_length", 160),
            n_fft=pre.get("n_fft", 320),
            center=pre.get("center", False),
            feat_in=enc.get("feat_in", 64),
            n_layers=enc.get("n_layers", 16),
            d_model=enc.get("d_model", 768),
            subsampling=enc.get("subsampling", "conv1d"),
            subs_kernel_size=enc.get("subs_kernel_size", 5),
            subsampling_factor=enc.get("subsampling_factor", 4),
            ff_expansion_factor=enc.get("ff_expansion_factor", 4),
            self_attention_model=enc.get("self_attention_model", "rotary"),
            pos_emb_max_len=enc.get("pos_emb_max_len", 5000),
            n_heads=enc.get("n_heads", 16),
            conv_kernel_size=enc.get("conv_kernel_size", 5),
            conv_norm_type=enc.get("conv_norm_type", "layer_norm"),
            head_type=d.get("head_type", "ctc"),
            num_classes=head.get("num_classes", 34),
            vocabulary=d.get("vocabulary"),
        )


# ─────────────────────── Audio preprocessing ───────────────────────

def mel_filters(sr: int, n_fft: int, n_mels: int) -> mx.array:
    """HTK mel filterbank (matching torchaudio default for GigaAM)."""
    def hz_to_mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)
    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    f_min, f_max = 0.0, sr / 2.0
    mel_min, mel_max = hz_to_mel(f_min), hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])

    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_fft // 2 + 1, n_mels), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        for k in range(lo, mid):
            if mid != lo:
                fb[k, i] = (k - lo) / (mid - lo)
        for k in range(mid, hi):
            if hi != mid:
                fb[k, i] = (hi - k) / (hi - mid)
    return mx.array(fb)


def hanning_window(size: int) -> mx.array:
    """Hanning window."""
    n = np.arange(size, dtype=np.float32)
    return mx.array(0.5 - 0.5 * np.cos(2.0 * np.pi * n / size))


def stft(signal: mx.array, n_fft: int, hop: int, win_len: int, window: mx.array) -> mx.array:
    """Simple STFT using mx operations."""
    # Pad if necessary
    pad_amount = n_fft // 2
    # For center=False, no padding
    length = signal.shape[-1]
    # Number of frames
    n_frames = 1 + (length - win_len) // hop

    # Build frames via strided indexing
    indices = mx.arange(win_len)[None, :] + (mx.arange(n_frames) * hop)[:, None]
    frames = signal[indices] * window[None, :]

    # Zero-pad to n_fft if needed
    if win_len < n_fft:
        pad_size = n_fft - win_len
        frames = mx.pad(frames, ((0, 0), (0, pad_size)))

    # Real FFT
    spectrum = mx.fft.rfft(frames)
    return spectrum  # [n_frames, n_fft//2 + 1]


def log_mel_spectrogram(audio: mx.array, cfg: GigaAMConfig,
                        mel_fb: Optional[mx.array] = None,
                        stft_win: Optional[mx.array] = None) -> mx.array:
    """Compute log-mel spectrogram matching GigaAM FeatureExtractor."""
    window = stft_win if stft_win is not None else hanning_window(cfg.win_length)
    spec = stft(audio, cfg.n_fft, cfg.hop_length, cfg.win_length, window)
    # Power spectrum
    power = mx.square(mx.abs(spec))  # [T, n_fft//2+1]
    # Mel filterbank
    if mel_fb is not None:
        filters = mel_fb
    else:
        filters = mel_filters(cfg.sample_rate, cfg.n_fft, cfg.features)  # [n_fft//2+1, n_mels]
    mel = power @ filters  # [T, n_mels]
    # Log
    log_mel = mx.log(mx.clip(mel, 1e-9, 1e9))
    return log_mel  # [T, n_mels]


# ─────────────────────── Model layers ───────────────────────

class Conv1dSubsampling(nn.Module):
    """Conv1d striding subsampling: 2 conv1d layers with stride=2, ReLU."""
    def __init__(self, cfg: GigaAMConfig):
        super().__init__()
        ks = cfg.subs_kernel_size
        pad = (ks - 1) // 2
        n_subs = int(math.log2(cfg.subsampling_factor))
        
        layers = []
        in_ch = cfg.feat_in
        for _ in range(n_subs):
            layers.append(nn.Conv1d(in_ch, cfg.d_model, kernel_size=ks, stride=2, padding=pad))
            layers.append(nn.ReLU())
            in_ch = cfg.d_model
        self.conv = layers
        self._n_subs = n_subs
        self._ks = ks
        self._pad = pad

    def __call__(self, x: mx.array, lengths: mx.array) -> Tuple[mx.array, mx.array]:
        """x: [B, T, feat_in] → [B, T', d_model]"""
        for layer in self.conv:
            x = layer(x)
        # Compute output lengths
        for _ in range(self._n_subs):
            lengths = mx.floor((lengths.astype(mx.float32) + 2 * self._pad - self._ks) / 2 + 1).astype(mx.int32)
        return x, lengths


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)."""
    def __init__(self, dim: int, base: int = 10000, max_len: int = 5000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        self._inv_freq = mx.array(inv_freq)

    def __call__(self, seq_len: int) -> Tuple[mx.array, mx.array]:
        t = mx.arange(seq_len).astype(mx.float32)
        freqs = mx.outer(t, self._inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


def rotate_half(x: mx.array) -> mx.array:
    """Rotates half the hidden dims of the input."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> Tuple[mx.array, mx.array]:
    """Apply rotary embeddings. q,k: [B, H, T, D], cos,sin: [T, D]"""
    cos = cos[None, None, :, :]  # [1, 1, T, D]
    sin = sin[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class RotaryMultiHeadAttention(nn.Module):
    """Multi-head attention with rotary positional embeddings.
    
    GigaAM applies RoPE BEFORE linear projections:
    1. Reshape raw input to multi-head: [B, T, D] → [B, T, H, d_k]
    2. Apply RoPE to query/key heads
    3. Reshape back to [B, T, D]
    4. Project through linear_q/k/v
    5. Standard scaled dot-product attention
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array,
                 mask: Optional[mx.array] = None) -> mx.array:
        B, T, D = x.shape
        H, d_k = self.n_heads, self.d_k

        # 1. Reshape raw input to multi-head for RoPE: [B, T, H, d_k]
        x_heads = x.reshape(B, T, H, d_k)

        # 2. Apply RoPE to query and key (same input for self-attention)
        # cos, sin: [T, d_k] → [1, T, 1, d_k]
        cos_e = cos[None, :, None, :]
        sin_e = sin[None, :, None, :]
        q_rot = x_heads * cos_e + rotate_half(x_heads) * sin_e
        k_rot = x_heads * cos_e + rotate_half(x_heads) * sin_e

        # 3. Reshape back to [B, T, D]
        q_rot = q_rot.reshape(B, T, D)
        k_rot = k_rot.reshape(B, T, D)

        # 4. Project through linear layers
        q = self.linear_q(q_rot).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)  # [B, H, T, d_k]
        k = self.linear_k(k_rot).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        v = self.linear_v(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)  # value uses original x

        # 5. Scaled dot-product attention
        scale = math.sqrt(d_k)
        attn = (q @ k.transpose(0, 1, 3, 2)) / scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v  # [B, H, T, d_k]
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.linear_out(out)


class ConformerConvolution(nn.Module):
    """Conformer convolution module with LayerNorm."""
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                         padding=pad, groups=d_model)
        # GigaAM v3 uses layer_norm for conv_norm_type
        self.batch_norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, D]
        x = self.pointwise_conv1(x)  # → [B, T, 2*D]
        # GLU: split last dim in half, apply sigmoid gate
        half = x.shape[-1] // 2
        x = x[..., :half] * mx.sigmoid(x[..., half:])  # GLU
        x = self.depthwise_conv(x)   # → [B, T, D]
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)  # → [B, T, D]
        return x


class ConformerFeedForward(nn.Module):
    """Conformer feed-forward module."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(self.activation(self.linear1(x)))


class ConformerLayer(nn.Module):
    """Single conformer layer."""
    def __init__(self, cfg: GigaAMConfig):
        super().__init__()
        d = cfg.d_model
        d_ff = d * cfg.ff_expansion_factor

        self.norm_feed_forward1 = nn.LayerNorm(d)
        self.feed_forward1 = ConformerFeedForward(d, d_ff)
        self.norm_self_att = nn.LayerNorm(d)
        self.self_attn = RotaryMultiHeadAttention(d, cfg.n_heads)
        self.norm_conv = nn.LayerNorm(d)
        self.conv = ConformerConvolution(d, cfg.conv_kernel_size)
        self.norm_feed_forward2 = nn.LayerNorm(d)
        self.feed_forward2 = ConformerFeedForward(d, d_ff)
        self.norm_out = nn.LayerNorm(d)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array,
                 mask: Optional[mx.array] = None) -> mx.array:
        # FF1
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + x * 0.5

        # Self-attention
        x = self.norm_self_att(residual)
        x = self.self_attn(x, cos, sin, mask=mask)
        residual = residual + x

        # Conv
        x = self.norm_conv(residual)
        x = self.conv(x)
        residual = residual + x

        # FF2
        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + x * 0.5

        return self.norm_out(residual)


class ConformerEncoder(nn.Module):
    """GigaAM Conformer encoder."""
    def __init__(self, cfg: GigaAMConfig):
        super().__init__()
        self.pre_encode = Conv1dSubsampling(cfg)
        self.pos_enc = RotaryPositionalEmbedding(
            cfg.d_model // cfg.n_heads,
            base=10000,
            max_len=cfg.pos_emb_max_len,
        )
        self.layers = [ConformerLayer(cfg) for _ in range(cfg.n_layers)]

    def __call__(self, features: mx.array, lengths: mx.array) -> Tuple[mx.array, mx.array]:
        """features: [B, T, feat_in] → encoded: [B, T', D], lengths: [B]"""
        x, lengths = self.pre_encode(features, lengths)
        T = x.shape[1]
        cos, sin = self.pos_enc(T)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x, lengths


class CTCHead(nn.Module):
    """CTC decoder head: Conv1d(d_model, num_classes, kernel=1)."""
    def __init__(self, feat_in: int, num_classes: int):
        super().__init__()
        self.decoder_layers = [nn.Conv1d(feat_in, num_classes, kernel_size=1)]

    def __call__(self, x: mx.array) -> mx.array:
        """x: [B, T, D] → log_probs: [B, T, C]"""
        logits = self.decoder_layers[0](x)  # [B, T, C]
        return mx.softmax(logits, axis=-1)  # we'll use log later


# ─────────────────────── Streaming dataclasses ───────────────────────

@dataclass
class StreamingConfig:
    """Configuration for pseudo-streaming transcription.

    Attributes:
        chunk_duration: Duration of each audio chunk in seconds.
        context_duration: Extra audio from before the chunk to keep as context.
            The model always sees [context + chunk] for better accuracy.
        step_duration: How much to advance between steps. If None, equals chunk_duration.
    """
    chunk_duration: float = 2.0
    context_duration: float = 3.0
    step_duration: Optional[float] = None

    def __post_init__(self):
        if self.step_duration is None:
            self.step_duration = self.chunk_duration


@dataclass
class StreamingResult:
    """Result from streaming transcription — mlx-audio compatible.

    Attributes:
        text: New text since last emission.
        is_final: True if this is the final result.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        progress: Progress 0.0–1.0.
        audio_position: Current position in audio (seconds).
        audio_duration: Total audio duration (seconds), 0 for live.
        cumulative_text: Full accumulated transcription so far.
        language: Language code.
    """
    text: str
    is_final: bool
    start_time: float
    end_time: float
    progress: float = 0.0
    audio_position: float = 0.0
    audio_duration: float = 0.0
    cumulative_text: str = ""
    language: str = "ru"


class GigaAMCTC(nn.Module):
    """Full GigaAM CTC model."""
    def __init__(self, cfg: GigaAMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ConformerEncoder(cfg)
        self.head = CTCHead(cfg.d_model, cfg.num_classes + 1)  # +1 for blank
        self.mel_filterbank: Optional[mx.array] = None
        self.stft_window: Optional[mx.array] = None

    def __call__(self, features: mx.array, lengths: mx.array) -> Tuple[mx.array, mx.array]:
        encoded, enc_lengths = self.encoder(features, lengths)
        log_probs = self.head(encoded)
        return log_probs, enc_lengths

    def _ctc_decode(self, log_probs: mx.array, enc_length: int) -> str:
        """CTC greedy decode: collapse repeated + remove blanks."""
        vocab = self.cfg.vocabulary
        blank_id = len(vocab)
        labels = mx.argmax(log_probs, axis=-1)
        labels_list = labels.tolist()

        result = []
        prev = -1
        for t, label in enumerate(labels_list):
            if t >= enc_length:
                break
            if label == blank_id:
                prev = label
                continue
            if label == prev:
                continue
            result.append(label)
            prev = label

        return "".join(vocab[i] for i in result)

    def _compute_features(self, audio: mx.array) -> Tuple[mx.array, mx.array]:
        """Audio → mel features [1, T, features] + lengths [1]."""
        mel = log_mel_spectrogram(audio, self.cfg,
                                   mel_fb=self.mel_filterbank,
                                   stft_win=self.stft_window)
        mel = mx.expand_dims(mel, 0)
        lengths = mx.array([mel.shape[1]])
        return mel, lengths

    def transcribe(self, audio: mx.array) -> str:
        """Transcribe raw audio waveform → text."""
        mel, lengths = self._compute_features(audio)
        log_probs, enc_lengths = self(mel, lengths)
        mx.eval(log_probs, enc_lengths)
        return self._ctc_decode(log_probs[0], int(enc_lengths[0]))

    def transcribe_chunk(self, audio: mx.array) -> str:
        """Transcribe a single chunk (for streaming). Same as transcribe but clearer name."""
        return self.transcribe(audio)

    def stream_generate(
        self,
        audio: mx.array,
        config: Optional[StreamingConfig] = None,
    ) -> Generator[StreamingResult, None, None]:
        """Pseudo-streaming transcription over pre-recorded audio.

        Uses growing buffer approach: each step transcribes from the start
        up to the current position. GigaAM at 85x realtime makes this fast
        even for 30s audio (~0.4s inference).

        For very long audio (>30s), falls back to sliding window of last 30s.

        Args:
            audio: Raw audio waveform, mx.array, 16kHz mono.
            config: StreamingConfig with step duration.

        Yields:
            StreamingResult with incremental text for each step.
        """
        if config is None:
            config = StreamingConfig()

        sr = self.cfg.sample_rate
        total_samples = audio.shape[0]
        audio_duration = total_samples / sr

        step_samples = int(config.step_duration * sr)
        max_window = int(30.0 * sr)  # cap at 30s for memory/speed

        previous_text = ""
        position = step_samples  # start with first step

        while position <= total_samples:
            is_last = position >= total_samples
            # Growing buffer from start, capped at 30s
            window_start = max(0, position - max_window)
            window = audio[window_start:position]

            current_text = self.transcribe(window)

            # Incremental text
            new_text = _incremental_text(previous_text, current_text)
            previous_text = current_text

            audio_pos = position / sr
            yield StreamingResult(
                text=new_text,
                is_final=is_last,
                start_time=window_start / sr,
                end_time=audio_pos,
                progress=position / total_samples,
                audio_position=audio_pos,
                audio_duration=audio_duration,
                cumulative_text=current_text,
                language="ru",
            )

            if is_last:
                break

            position = min(position + step_samples, total_samples)

    def stream_live(
        self,
        audio_buffer: mx.array,
    ) -> StreamingResult:
        """Transcribe a growing audio buffer for live microphone use.

        Call this repeatedly as new audio arrives. Transcribes the full buffer
        (capped at 30s from the end). GigaAM is 85x realtime so this is fast.

        Args:
            audio_buffer: The full accumulated audio so far.

        Returns:
            StreamingResult with current full transcription.
        """
        sr = self.cfg.sample_rate
        total_samples = audio_buffer.shape[0]
        max_window = int(30.0 * sr)

        start = max(0, total_samples - max_window)
        window = audio_buffer[start:]

        text = self.transcribe(window)

        return StreamingResult(
            text=text,
            is_final=False,
            start_time=start / sr,
            end_time=total_samples / sr,
            progress=0.0,
            audio_position=total_samples / sr,
            audio_duration=0.0,
            cumulative_text=text,
            language="ru",
        )


def _incremental_text(previous: str, current: str) -> str:
    """Find new text added to current vs previous.

    If current starts with previous → return the new suffix.
    If model corrected → return full current (with marker).
    """
    if not previous:
        return current
    if current.startswith(previous):
        return current[len(previous):]
    # Model self-corrected — return full updated text
    return current


def load_model(model_dir: str) -> GigaAMCTC:
    """Load converted GigaAM MLX model from directory."""
    model_dir = Path(model_dir)
    cfg = GigaAMConfig.from_file(str(model_dir / "config.json"))
    model = GigaAMCTC(cfg)
    weights = mx.load(str(model_dir / "model.safetensors"))
    
    # Extract preprocessing weights
    mel_fb = weights.pop("mel_filterbank", None)
    stft_win = weights.pop("stft_window", None)
    
    if mel_fb is not None:
        model.mel_filterbank = mel_fb.astype(mx.float32)
    if stft_win is not None:
        model.stft_window = stft_win.astype(mx.float32)
    
    # Load model weights
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


def load_audio(path: str, sample_rate: int = 16000) -> mx.array:
    """Load audio via ffmpeg → mx.array."""
    import subprocess
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-"
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    audio_np = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return mx.array(audio_np)
