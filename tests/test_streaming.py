import copy

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

from gigaam.decoder import RNNTHead
from gigaam.decoding import RNNTGreedyDecoding
from gigaam.model import GigaAMASR, GigaAMDualASR
from gigaam.streaming import DecoderStreamState, GigaAMStreaming


def tiny_model(before=True, center=False, subsampling="conv1d"):
    torch.manual_seed(2)
    cfg = OmegaConf.create(f"""
model_name: tiny_ctc
preprocessor:
  _target_: gigaam.preprocess.FeatureExtractor
  sample_rate: 800
  features: 8
  n_fft: 32
  win_length: 32
  hop_length: 8
  center: {center}
encoder:
  _target_: gigaam.encoder.ConformerEncoder
  feat_in: 8
  n_layers: 2
  d_model: 8
  subsampling: {subsampling}
  n_heads: 2
  conv_kernel_size: 3
  pos_emb_max_len: 128
  apply_rope_before_projections: {before}
  use_causal_conv_chunking: true
  subs_kernel_size: 3
head: {{_target_: gigaam.decoder.CTCHead, feat_in: 8, num_classes: 4}}
decoding: {{_target_: gigaam.decoding.CTCGreedyDecoding, vocabulary: [a, " ", b]}}
streaming:
  contexts:
    - {{future_context_ms: 160, left_context: 32}}
    - {{future_context_ms: 320, left_context: 32}}
""")
    return GigaAMASR(cfg).eval()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"center": True}, "center=False"),
        ({"subsampling": "conv2d"}, "Conv1d"),
    ],
)
def test_streaming_rejects_unsupported_frontend(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GigaAMStreaming(tiny_model(**kwargs), 160)


def dual_model(single):
    cfg = single.cfg.copy()
    cfg.heads = {name: cfg.head for name in ("e2e", "raw")}
    cfg.decodings = {name: cfg.decoding for name in cfg.heads}
    return GigaAMDualASR(cfg).eval()


def test_transcribe_rejects_long_audio_for_single_and_dual_heads(tmp_path):
    single = tiny_model()
    models = (single, dual_model(single))
    audio = tmp_path / "long.wav"
    sf.write(audio, np.zeros(30 * 16000 + 1, dtype=np.float32), 16000)

    for model in models:
        with pytest.raises(ValueError, match="transcribe_longform"):
            model.transcribe(str(audio))


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("timestamps", [False, True])
def test_offline_confidence(tmp_path, monkeypatch, dual, timestamps):
    model = tiny_model()
    if dual:
        model = dual_model(model)
    heads = model.heads if dual else {"e2e": model.head}
    expected = {}
    for i, (name, head) in enumerate(heads.items()):
        # Distinct, constant posteriors let us check that each head keeps its score.
        conv = head.decoder_layers[0]
        with torch.no_grad():
            conv.weight.zero_()
            conv.bias.zero_()
            conv.bias[2 * i] = i + 1
        expected[name] = conv.bias.softmax(0).max().item()
    wave = torch.zeros(1600)
    path = str(tmp_path / "short.wav")
    sf.write(path, wave.numpy(), 16000)
    boundaries = [(0.0, 0.03), (0.05, 0.08)]
    monkeypatch.setattr(
        "gigaam.vad_utils.segment_audio_file",
        lambda *args, **kwargs: ([wave[:480], wave[800:1280]], boundaries),
    )
    for options in ({}, {"return_confidence": True}):
        short = model.transcribe(path, word_timestamps=timestamps, **options)
        long = model.transcribe_longform(path, word_timestamps=timestamps, **options)
        for name, probability in expected.items():
            utterance = short[name] if dual else short
            segments = (long[name] if dual else long).segments
            assert [(s.start, s.end) for s in segments] == boundaries
            score = pytest.approx(probability) if options else None
            for item, start in zip([utterance, *segments], [0.0, 0.0, 0.05]):
                assert item.text == ("b" if name == "raw" else "a")
                assert item.confidence == score
                if timestamps:
                    (word,) = item.words
                    assert word.confidence == score and word.start == start
                    assert word.end > word.start
                else:
                    assert item.words is None


@pytest.mark.parametrize("timestamps", [False, True])
def test_stream_confidence_for_new_tokens(timestamps):
    stream = GigaAMStreaming(
        tiny_model(), 160, word_timestamps=timestamps, return_confidence=True
    )
    wave = torch.randn(1171)
    pieces = [stream.transcribe_partial(packet) for packet in wave.split(37)]
    pieces.append(stream.flush())
    scores = stream.decoder_state.token_logprobs
    offset = 0
    assert sum(bool(p.text) for p in pieces) > 1
    for piece in pieces:
        # In this character tokenizer every published character is one token.
        end = offset + len(piece.text)
        expected = (
            pytest.approx(np.exp(np.mean(scores[offset:end]))) if piece.text else None
        )
        assert piece.confidence == expected
        offset = end
    assert offset == len(scores) and stream.flush().confidence is None
    stream.reset_state()
    assert stream.decoder_state.token_logprobs == []
    assert stream.transcribe_partial(wave, final=True).confidence == pytest.approx(
        np.exp(np.mean(scores))
    )


@pytest.mark.parametrize("before", [True, False])
def test_packet_inference(before):
    model = tiny_model(before)
    wave = torch.randn(1171)
    whole = GigaAMStreaming(model, 160, word_timestamps=True)
    expected = whole.transcribe_partial(wave, final=True)
    stream = GigaAMStreaming(model, 160, word_timestamps=True)
    pieces = [stream.transcribe_partial(packet) for packet in wave.split(37)]
    assert any(piece.words for piece in pieces)
    published = copy.deepcopy([piece.words for piece in pieces])
    pieces.append(stream.flush())
    assert [piece.words for piece in pieces[:-1]] == published
    assert "".join(piece.text for piece in pieces) == expected.text
    words = [word for piece in pieces for word in piece.words]
    assert words == expected.words and len(words) > 1
    assert [word.start for word in words] == sorted(word.start for word in words)
    assert words[-1].end <= len(wave) / stream.sample_rate
    assert stream.decoder_state.tokens == whole.decoder_state.tokens
    assert stream.decoder_state.frames == whole.decoder_state.frames
    stream.reset_state()
    assert stream.transcribe_partial(wave, final=True) == expected


@pytest.mark.parametrize("chunks", [1, 2, 3, 7, 10])
def test_encoder_batches_ready_chunks(chunks):
    encoder = tiny_model().encoder
    chunk = 16
    features = torch.randn(1, 8, chunks * 64 + 20)
    length = torch.tensor([features.shape[-1]])
    state = encoder.create_stream_state(chunk, 32)
    calls = []
    hook = encoder.frontend.register_forward_hook(lambda *_: calls.append(1))
    first, _, state = encoder(features, length, state=state)
    assert len(calls) == 1 and state.pending.shape[-1] == 20
    tail, _, state = encoder(torch.empty(1, 8, 0), torch.tensor([0]), state, final=True)
    assert len(calls) == 2
    hook.remove()
    reference = encoder.create_stream_state(chunk, 32)
    pieces = [
        encoder(features[:, :, i * 64 : (i + 1) * 64], torch.tensor([64]), reference)[0]
        for i in range(chunks)
    ]
    pieces.append(
        encoder(features[:, :, -20:], torch.tensor([20]), reference, final=True)[0]
    )
    assert torch.allclose(
        torch.cat((first, tail), -1), torch.cat(pieces, -1), atol=1e-6
    )


@pytest.mark.parametrize("chunks", [1, 2, 3, 7, 10])
def test_attention_empty_history(chunks):
    encoder = tiny_model().encoder
    layer = encoder.layers[0]
    with torch.no_grad():
        for parameter in layer.feed_forward1.parameters():
            parameter.zero_()
    x = torch.randn(1, chunks * 4, 8)
    encoder.pos_enc.extend_pe(encoder.pos_emb_max_len, x.device)
    x[:, ::4] = 0
    x[:, 1::4] *= 1e-9
    valid = torch.ones(1, x.shape[1], dtype=torch.bool)
    valid[:, -1] = False
    cached = torch.zeros(1, 8, 8)
    cached[:, 0] = 1e-6
    cached[:, 1] = -1e-6
    cached[:, -2:] = -torch.arange(1, 17).reshape(1, 2, 8)

    def state(count):
        stream = encoder.create_stream_state(4, 8)
        stream.position, stream.valid = 8, torch.tensor(count)
        stream.layers[0].attention = cached.clone()
        return stream

    def run(values, mask, stream):
        return encoder._stream_layer(layer, values, stream.layers[0], stream, mask)

    with torch.inference_mode():
        expected = run(x, valid, state(2))
        torch.testing.assert_close(run(x, valid, state(8)), expected)
        stream, pieces = state(8), []
        for start in range(0, x.shape[1], 4):
            pieces.append(
                run(x[:, start : start + 4], valid[:, start : start + 4], stream)
            )
            stream.position += 4
            stream.valid = (stream.valid + 4).clamp_max(8)
        torch.testing.assert_close(torch.cat(pieces, 1), expected)
        stream = state(2)
        stream.layers[0].attention[:, :6] = 100
        torch.testing.assert_close(run(x, valid, stream), expected)
        stream = state(8)
        stream.layers[0].attention[:, -2:] = 0
        assert not torch.allclose(run(x, valid, stream), expected)


def tiny_rnnt_head():
    torch.manual_seed(7)
    head = RNNTHead(
        dict(pred_hidden=2, pred_rnn_layers=1, num_classes=3),
        dict(enc_hidden=2, pred_hidden=2, joint_hidden=2, num_classes=3),
    ).eval()
    with torch.no_grad():
        head.joint.enc.weight.copy_(torch.eye(2))
        head.joint.enc.bias.zero_()
        head.joint.pred.weight.zero_()
        head.joint.pred.bias.zero_()
        head.joint.joint_net[1].weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        )
        head.joint.joint_net[1].bias.zero_()
    return head


@pytest.mark.parametrize("windowed", [True, False])
def test_rnnt_scores_follow_emissions(windowed):
    head = tiny_rnnt_head()
    head.joint.supports_frame_windows = windowed
    decoder = RNNTGreedyDecoding(["a", " "], max_symbols_per_step=2)
    signal = torch.tensor(
        [[-2.0, -2.0, 1.0, -2.0, -2.0, 3.0], [4.0, -2.0, -2.0, 2.0, -2.0, 3.0]]
    )
    encoded = torch.stack((signal, -signal), dim=1)
    lengths = torch.tensor([6, 4])
    scores = []
    result = decoder.decode(head, encoded, lengths, token_logprobs=scores)
    assert result == decoder.decode(head, encoded, lengths)
    assert [item[2] for item in result] == [[2, 2, 5, 5], [0, 0, 3, 3]]
    # The fixture's joint is independent of prediction history. Each active
    # frame emits "a" twice before the symbol cap advances the decoder.
    for row, frames in enumerate(([2, 2, 5, 5], [0, 0, 3, 3])):
        features = encoded[row : row + 1].transpose(1, 2)
        prediction = torch.zeros(1, 1, 2)
        frame_scores = head.joint.joint(features, prediction)[0, :, 0, 0]
        assert scores[row] == pytest.approx(frame_scores[frames].detach().tolist())


def test_rnnt_predictor_cache_mixed_rows_and_packets():
    head = tiny_rnnt_head()
    decoding = RNNTGreedyDecoding(["a", " "], max_symbols_per_step=2)
    signal = torch.tensor([[-2.0, -2.0, 2.0, -2.0, -2.0], [2.0, -2.0, -2.0, 2.0, -2.0]])
    encoded = torch.stack((signal, -signal), dim=1)
    lengths = torch.tensor([5, 4])
    whole = [DecoderStreamState(), DecoderStreamState()]
    expected = decoding.decode(head, encoded, lengths, whole)
    assert expected == [("aa", [0, 0], [2, 2]), ("aaaa", [0] * 4, [0, 0, 3, 3])]
    packets = [DecoderStreamState(), DecoderStreamState()]
    calls = []
    hook = head.decoder.lstm.register_forward_hook(
        lambda module, inputs, output: calls.append(inputs[0].shape[1])
    )
    for start, end in [(0, 2), (2, 2), (2, 3), (3, 5)]:
        decoding.decode(
            head,
            encoded[:, :, start:end],
            (lengths - start).clamp(0, end - start),
            packets,
        )
    hook.remove()
    assert sum(calls) == 7
    for actual, reference in zip(packets, whole):
        assert (actual.tokens, actual.frames, actual.previous, actual.frame_offset) == (
            reference.tokens,
            reference.frames,
            reference.previous,
            reference.frame_offset,
        )
        for actual_part, reference_part in zip(actual.hidden, reference.hidden):
            torch.testing.assert_close(actual_part, reference_part)
    reset = [DecoderStreamState(), DecoderStreamState()]
    assert decoding.decode(head, encoded, lengths, reset) == expected


def test_rnnt_predictor_cache_all_blank():
    head = tiny_rnnt_head()
    decoding = RNNTGreedyDecoding(["a", " "])
    encoded = torch.tensor([[[-2.0] * 12, [2.0] * 12]])
    state, calls = DecoderStreamState(), []
    hook = head.decoder.lstm.register_forward_hook(lambda *_: calls.append(1))
    for packet in encoded.split(3, dim=-1):
        assert decoding.decode(head, packet, torch.tensor([3]), [state]) == [
            ("", [], [])
        ]
    hook.remove()
    assert len(calls) == 1
    assert state.hidden is None and state.previous is None and state.frame_offset == 12
