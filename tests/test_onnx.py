import logging

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_language import tiny_conditioned
from test_streaming import tiny_model

import gigaam
from gigaam.decoding import Tokenizer
from gigaam.model import GigaAMDualASR
from gigaam.onnx_utils import OnnxGigaAMStreaming, _Model, infer_onnx, load_onnx
from gigaam.streaming import GigaAMStreaming
from gigaam.utils import _namespace_onnx_external_data, download_short_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_audio():
    """Provide test audio file for all tests"""
    return download_short_audio()


def test_ctc_one_pass_uses_both_heads_at_each_context(tmp_path):
    cfg = tiny_conditioned().cfg
    cfg.model_name = "v4_ctc"
    del cfg.language_conditioning
    for name in cfg.heads:
        cfg.heads[name] = cfg.head.copy()
        cfg.decodings[name] = cfg.decoding.copy()
    model = GigaAMDualASR(cfg).eval()
    prefix = model.to_onnx(tmp_path, streaming=True, future_context_ms=160)
    sessions, cfg = load_onnx(tmp_path, prefix)
    waves = [torch.randn(size) for size in (1267, 1139)]
    assert set(sessions["fc"]) == {"raw", "e2e"}
    for context in (None, 160, 320):
        streaming = context is not None
        output = infer_onnx(
            waves,
            cfg,
            sessions,
            batch_size=2,
            progress=False,
            streaming=streaming,
            future_context_ms=context,
        )
        for i, wave in enumerate(waves):
            stream = OnnxGigaAMStreaming(
                cfg, sessions, future_context_ms=context, streaming=streaming
            )
            expected = stream.transcribe_partial(wave, final=True)
            assert {name: rows[i] for name, rows in output.items()} == {
                name: result.text for name, result in expected.items()
            }


@pytest.mark.parametrize(
    "revision, export_dtype",
    [
        ("emo", torch.float32),
        ("v2_ssl", torch.float16),
        ("v3_ctc", torch.float16),
        ("v3_e2e_rnnt", torch.float32),
    ],
)
def test_onnx_converting(revision, export_dtype, test_audio, tmp_path):
    """Test model revision converts to ONNX and produces correct batched output."""
    onnx_dir = str(tmp_path)
    model = gigaam.load_model(revision)
    model.to_onnx(dir_path=onnx_dir, dtype=export_dtype)
    sessions, model_cfg = load_onnx(onnx_dir, revision)

    data = [test_audio, test_audio]
    result = infer_onnx(data, model_cfg, sessions, batch_size=2)

    assert isinstance(result, list), f"{revision}: expected list, got {type(result)}"
    assert len(result) == 2, f"{revision}: expected 2 results, got {len(result)}"

    if "ssl" in revision:
        for i, r in enumerate(result):
            assert isinstance(r, np.ndarray), f"{revision}[{i}]: expected ndarray"
            assert r.ndim == 2, f"{revision}[{i}]: expected 2D array, got {r.ndim}D"

        orig_embed = model.embed_audio(test_audio)[0].detach().cpu().numpy()
        tol = 0.01 if export_dtype == torch.float16 else 0.001
        for i in range(2):
            diff = np.abs(orig_embed - result[i]).mean()
            assert diff < tol, f"{revision}[{i}]: ONNX embed diff {diff}"

    elif "emo" in revision:
        orig_probs = model.get_probs(test_audio)
        tol = 1e-3 if export_dtype == torch.float16 else 1e-4
        for i in range(2):
            r = result[i]
            assert isinstance(r, np.ndarray), f"{revision}[{i}]: expected ndarray"
            pred_probs = {
                model.id2name[j]: float(r[j]) for j in range(len(model.id2name))
            }
            assert all(
                abs(orig_probs[em] - pred_probs[em]) < tol for em in orig_probs
            ), f"{revision}[{i}]: ONNX emo probs failed: {pred_probs}"

    else:
        orig_text = model.transcribe(test_audio).text
        for i in range(2):
            assert isinstance(result[i], str), f"{revision}[{i}]: expected str"
            assert (
                orig_text == result[i]
            ), f"{revision}[{i}]: ONNX transcribe failed: {result[i]}"


@pytest.mark.parametrize("kind", ["conformer", "zapformer"])
def test_dynamic_export_and_packet_inference(tmp_path, kind):
    model = tiny_conditioned()
    language = "ru-en"
    if kind == "conformer":
        cfg = tiny_model(before=False).cfg
        cfg.heads = {name: cfg.head for name in ("raw", "e2e")}
        cfg.decodings = {name: cfg.decoding for name in cfg.heads}
        cfg.streaming.contexts.append(dict(future_context_ms=640, left_context=32))
        model, language = GigaAMDualASR(cfg).eval(), None
    context = 640 if kind == "conformer" else 160
    prefix = GigaAMStreaming(model, context, language=language).to_onnx(tmp_path)
    sessions, cfg = load_onnx(tmp_path, prefix)
    offline_sessions, offline_cfg = load_onnx(tmp_path, prefix, streaming=False)
    # Leave more padding than the Conformer attention window for the shorter clip.
    wave = torch.randn(2579)
    with torch.inference_mode():
        encoded, lengths = model(wave[None], torch.tensor([len(wave)]), language)
        expected = model.decode(encoded, lengths)
    assert (
        infer_onnx(
            [wave],
            offline_cfg,
            offline_sessions,
            language=language,
            streaming=False,
            progress=False,
        )
        == expected
    )

    if kind == "zapformer":
        fixed_dir = tmp_path / "fixed"
        fixed_prefix = GigaAMStreaming(model, context, language=language).to_onnx(
            fixed_dir, dynamic_context=False
        )
        fixed_sessions, fixed_cfg = load_onnx(fixed_dir, fixed_prefix)
        with pytest.raises(ValueError, match="exported context"):
            OnnxGigaAMStreaming(
                fixed_cfg,
                fixed_sessions,
                future_context_ms=320,
                language=language,
            )
        fixed = OnnxGigaAMStreaming(fixed_cfg, fixed_sessions, language=language)
        with pytest.raises(ValueError, match="exported context"):
            fixed.set_future_context_ms(320)
        with pytest.raises(ValueError, match="exported context"):
            OnnxGigaAMStreaming(fixed_cfg, fixed_sessions, streaming=False)
    for streaming in (True, False):
        actual = OnnxGigaAMStreaming(
            cfg,
            sessions,
            future_context_ms=context if streaming else None,
            streaming=streaming,
            language=language,
        )
        expected_audio, actual_audio = [], []
        hook = actual.model.encoder.register_forward_hook(
            lambda module, inputs, output: actual_audio.append(output[0].detach())
        )
        if streaming:
            expected = GigaAMStreaming(model, context, language=language)

            def capture(module, inputs, output):
                encoded = output[0]
                if model.language_fusion is not None:
                    encoded = model.language_fusion(encoded, expected._lang_embedding)
                expected_audio.append(encoded.detach())

            native_hook = model.encoder.register_forward_hook(capture)
            text = {
                name: result.text
                for name, result in expected.transcribe_partial(
                    wave, final=True
                ).items()
            }
            native_hook.remove()
        else:
            with torch.inference_mode():
                encoded, lengths = model(
                    wave[None], torch.tensor([len(wave)]), language
                )
                text = {
                    name: value[0]
                    for name, value in model.decode(encoded, lengths).items()
                }
                expected_audio.append(encoded)
        result = dict(raw="", e2e="")
        for packet in wave.split(97):
            for name, value in actual.transcribe_partial(packet).items():
                result[name] += value.text
        for name, value in actual.flush().items():
            result[name] += value.text
        hook.remove()
        assert result == text
        if not streaming:
            offline = tmp_path / "offline"
            model.to_onnx(offline)
            offline_sessions, offline_cfg = load_onnx(offline, model.cfg.model_name)
            assert infer_onnx(
                [wave], offline_cfg, offline_sessions, language=language, progress=False
            ) == {name: [value] for name, value in text.items()}
        batch = torch.stack(
            (wave, torch.nn.functional.pad(wave[:659], (0, len(wave) - 659)))
        )
        with torch.inference_mode():
            encoded, lengths = model(
                batch,
                torch.tensor([len(wave), 659]),
                language,
                future_context_ms=context if streaming else None,
            )
            for index, length in enumerate((len(wave), 659)):
                single, valid = model(
                    batch[index : index + 1, :length],
                    torch.tensor([length]),
                    language,
                    future_context_ms=context if streaming else None,
                )
                torch.testing.assert_close(
                    encoded[index, :, : valid[0]], single[0], atol=2e-5, rtol=2e-4
                )
        assert infer_onnx(
            [wave, wave[:659]],
            cfg,
            sessions,
            language=language,
            streaming=streaming,
            future_context_ms=context if streaming else None,
            progress=False,
        ) == model.decode(encoded, lengths)
        torch.testing.assert_close(
            torch.cat(actual_audio, -1),
            torch.cat(expected_audio, -1),
            atol=2e-5,
            rtol=2e-4,
        )


def test_supplied_tokenizer_skips_missing_onnx_tokenizer_file():
    class Session:
        def get_inputs(self):
            return []

    cfg = OmegaConf.create(
        {
            "model_name": "v3_e2e_rnnt",
            "decoding": {
                "_target_": "gigaam.decoding.RNNTGreedyDecoding",
                "vocabulary": [],
                "model_path": "/missing/exporter/tokenizer.model",
            },
            "head": {},
        }
    )
    tokenizer = Tokenizer(["a"])
    model = _Model(cfg, [Session()], tokenizer=tokenizer)
    assert model.decodings["head"].tokenizer is tokenizer
    assert model.decodings["head"].blank_id == len(tokenizer)


def test_explicit_model_ignores_unrelated_directory_manifest(tmp_path):
    requested = OmegaConf.create(
        {"model_name": "requested_ctc", "heads": {}, "decodings": {}}
    )
    unrelated = OmegaConf.create(
        {"model_name": "unrelated_rnnt", "heads": {}, "decodings": {}}
    )
    OmegaConf.save(requested, tmp_path / "requested_ctc.yaml")
    OmegaConf.save(unrelated, tmp_path / "unrelated_rnnt.yaml")
    (tmp_path / "streaming.json").write_text(
        '{"model_name":"unrelated_rnnt","step":{"file":"step.onnx"},'
        '"frontend":{"file":"front.onnx","states":[]}}',
        encoding="utf-8",
    )
    sessions, cfg = load_onnx(
        tmp_path, "requested_ctc", session_factory=lambda path: path.name
    )
    assert cfg.model_name == "requested_ctc"
    assert sessions[0] == "requested_ctc.onnx"


def test_external_weights_are_namespaced(tmp_path):
    import onnx
    from onnx import helper, numpy_helper

    for size in (1, 2):
        weight = numpy_helper.from_array(np.full(size, size, dtype=np.float32))
        graph = helper.make_graph([], "weights", [], [], [weight])
        path = tmp_path / f"graph{size}.onnx"
        onnx.save_model(
            helper.make_model(graph),
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="shared",
            size_threshold=0,
        )
        _namespace_onnx_external_data(path)
    for size in (1, 2):
        graph = onnx.load(tmp_path / f"graph{size}.onnx").graph
        np.testing.assert_array_equal(
            numpy_helper.to_array(graph.initializer[0]), np.full(size, size)
        )


@pytest.mark.parametrize("provider", ["CPUExecutionProvider", "CUDAExecutionProvider"])
def test_mixed_cache_output_is_normalized_for_next_packet(tmp_path, provider):
    import onnxruntime as ort

    from gigaam.onnx_utils import _Encoder

    if provider not in ort.get_available_providers():
        pytest.skip(f"{provider} is unavailable")

    class CacheStep(torch.nn.Module):
        def forward(self, features, lengths, position, valid, cache):
            return features, lengths, position, valid, cache.float() + 1

    features, lengths = torch.ones(1, 2, 4), torch.tensor([4])
    path = tmp_path / "mixed_cache.onnx"
    cache, position = torch.zeros(1, 2, dtype=torch.float16), torch.tensor([0])
    inputs = (features, lengths, position, position.clone(), cache)
    torch.onnx.export(
        CacheStep(),
        inputs,
        str(path),
        input_names=["features", "lengths", "position", "valid", "cache"],
        output_names=["encoded", "lengths", "position", "valid", "next_cache"],
        opset_version=17,
        dynamo=False,
    )
    manifest = dict(
        future_context_ms=4,
        frame_shift_ms=1,
        subsampling_factor=1,
        encoder_dim=2,
        ctc_labels=["raw", "e2e"],
        frontend={"states": []},
        states=[{"shape": [1, 2], "dtype": "float16"}],
    )
    session = ort.InferenceSession(str(path), providers=[provider])
    encoder = _Encoder(manifest, {"encoder": session})
    state = encoder.create_stream_state(4, 0)
    for _ in range(2):
        encoder(features, lengths, state)
    actual = state.caches[0]
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()
    assert actual.dtype == np.float16
    np.testing.assert_array_equal(actual, np.full((1, 2), 2, dtype=np.float16))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
