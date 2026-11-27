from concurrent.futures import ThreadPoolExecutor

import soundfile as sf
import torch
from omegaconf import OmegaConf
from test_streaming import tiny_model

from gigaam.language import LanguageEmbedding
from gigaam.model import GigaAMDualASR
from gigaam.streaming import GigaAMStreaming


def tiny_conditioned():
    cfg = tiny_model().cfg
    cfg.model_name = "tiny_zapformer_lstm"
    cfg.encoder_autocast_dtype = "float32"
    cfg.encoder = OmegaConf.create("""
_target_: gigaam.encoder.ZapformerEncoder
in_channels: 8
frontend_channels: 2,2,4
embed_dim: 8
encoder_dim: 8,8
downsampling_factor: 1,2
num_encoder_layers: 1,1
num_heads: 2,2
query_head_dim: 4,4
value_head_dim: 4,4
feedforward_multiple: 2,2
conv_params: 3,3
causal_frontend: true
dual_mode: true
chunk_size: [-1, 4, 8, 16]
window_size: [[-1, -1], [32, 0], [32, 0], [32, 0]]
max_seq_len: 128
""")
    cfg.language_conditioning = OmegaConf.create("""
languages: [ru, en, kk]
num_slots: 8
embedding_dim: 4
film: {enabled: true, placement: all}
late_fusion: {enabled: true, fusion: mlp, hidden: 12, activation: relu}
""")
    cfg.heads, cfg.decodings = {}, {}
    cfg.streaming.contexts.append(dict(future_context_ms=640, left_context=32))
    for name, labels in (("raw", ["a", "b"]), ("e2e", ["A", "B", "C"])):
        cfg.heads[name] = dict(
            _target_="gigaam.decoder.RNNTHead",
            decoder=dict(
                pred_hidden=16, num_classes=len(labels) + 1, pred_rnn_layers=1
            ),
            joint=dict(
                enc_hidden=8,
                pred_hidden=16,
                joint_hidden=8,
                num_classes=len(labels) + 1,
            ),
        )
        cfg.decodings[name] = dict(
            _target_="gigaam.decoding.RNNTGreedyDecoding",
            vocabulary=labels,
            max_symbols_per_step=3,
        )
    model = GigaAMDualASR(cfg).eval()
    with torch.no_grad():
        for name, token in (("raw", 0), ("e2e", 1)):
            output = model.heads[name].joint.joint_net[-1]
            output.weight.zero_()
            output.bias.fill_(-10)
            output.bias[token] = 10
    return model


def test_language_mixtures_and_default_auto():
    language = LanguageEmbedding(
        dict(languages=["ru", "en"], num_slots=3, embedding_dim=4)
    )
    torch.testing.assert_close(language("ru-en"), (language("ru") + language("en")) / 2)
    torch.testing.assert_close(language(None), language("auto"))


def test_zapformer_context_modes_are_request_local():
    model = tiny_conditioned()
    audio = torch.randn(1, 3200)
    lengths = torch.tensor([audio.shape[-1]])

    def forward(context):
        with torch.inference_mode():
            return model(audio, lengths, "ru-en", future_context_ms=context)

    expected = {context: forward(context) for context in (None, 160)}
    requests = [None, 160] * 8
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(forward, requests))

    for context, (actual, actual_lengths) in zip(requests, outputs):
        reference, reference_lengths = expected[context]
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual_lengths, reference_lengths)


def test_dual_packets(tmp_path):
    model = tiny_conditioned()
    wave = torch.randn(771)
    whole = GigaAMStreaming(model, 160, language="ru-en")
    expected = {
        name: result.text
        for name, result in whole.transcribe_partial(wave, final=True).items()
    }
    stream = GigaAMStreaming(model, 160, language="ru-en")
    text = dict(raw="", e2e="")
    for packet in [*wave.split(29), torch.empty(0)]:
        for name, value in stream.transcribe_partial(
            packet, final=not len(packet)
        ).items():
            text[name] += value.text
    assert text == expected
    assert set(text["raw"]) == {"a"} and set(text["e2e"]) == {"B"}
    for name in text:
        assert stream.decoder_states[name].tokens == whole.decoder_states[name].tokens
        torch.testing.assert_close(
            stream.decoder_states[name].hidden, whole.decoder_states[name].hidden
        )
    assert (
        stream.decoder_states["raw"].hidden is not stream.decoder_states["e2e"].hidden
    )
    path = tmp_path / "audio.wav"
    sf.write(path, wave.numpy(), 16000, subtype="FLOAT")
    result = model.transcribe(str(path), True, language="ru-en")
    assert {name: value.text for name, value in result.items()} == expected
    assert all(
        value.words and value.words[-1].end <= len(wave) / 16000
        for value in result.values()
    )
