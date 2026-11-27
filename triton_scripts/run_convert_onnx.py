import argparse
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gigaam  # noqa: E402

BACKENDS = {
    "CPUExecutionProvider": "cpu",
    "CUDAExecutionProvider": "cuda",
    "TensorRT": "trt",
    "TensorRT-fp16": "trt_fp16",
}


def convert_dual_repository(
    model, contexts, repository, provider: str, dynamic_context=None, max_streams=32
) -> None:
    """Generate isolated dual repositories without changing legacy ensembles."""
    from pathlib import Path

    backend = BACKENDS[provider]
    template = Path(__file__).with_name("dual_backend.py")
    if dynamic_context is None:
        dynamic_context = bool(contexts)
    saved_contexts = (
        {int(c.future_context_ms) for c in model.cfg.streaming.contexts}
        if dynamic_context or contexts
        else set()
    )
    default_ms = max(contexts or saved_contexts) if dynamic_context else None
    for milliseconds in [default_ms] if dynamic_context else [None, *contexts]:
        mode = "offline" if milliseconds is None else "cached"
        suffix = (
            "dynamic"
            if dynamic_context
            else ("offline" if milliseconds is None else f"stream{milliseconds}")
        )
        name = f"{model.cfg.model_name}_dual_{suffix}_{backend}"
        directory = Path(repository) / name
        version = directory / "1"
        graphs = version / "graphs"
        graphs.mkdir(parents=True)
        model.encoder._triton_batch = milliseconds is not None and hasattr(
            model.encoder, "downsampling_factor"
        )
        model.encoder._triton_native = provider == "TensorRT-fp16"
        model.encoder._triton_provider = provider
        try:
            exported_prefix = model.to_onnx(
                str(graphs),
                dtype=torch.float32,
                future_context_ms=milliseconds,
                streaming=milliseconds is not None,
                dynamic_context=bool(dynamic_context),
            )
        finally:
            del model.encoder._triton_batch
            del model.encoder._triton_native
            del model.encoder._triton_provider
        shutil.copyfile(template, version / "model.py")
        if provider in ("TensorRT", "TensorRT-fp16"):
            shutil.copyfile(
                template.with_name("trt_utils.py"), version / "trt_utils.py"
            )
        inputs = [
            ("AUDIO", "FP32", -1),
            ("AUDIO_LENGTH", "INT64", -1),
            ("LANGUAGE", "STRING", -1),
        ]
        if milliseconds is not None:
            inputs += [
                ("STREAM_ID", "STRING", -1),
                ("START", "BOOL", -1),
                ("END", "BOOL", -1),
            ]
            if dynamic_context:
                inputs += [
                    ("STREAMING", "BOOL", -1),
                    ("FUTURE_CONTEXT_MS", "INT64", -1),
                ]
        outputs = [(head + "_TEXT", "STRING", -1) for head in ("RAW", "E2E")]

        def tensor(name, dtype, size):
            optional = (
                " optional: true"
                if name in ("LANGUAGE", "STREAMING", "FUTURE_CONTEXT_MS")
                else ""
            )
            return (
                f'{{ name: "{name}" data_type: TYPE_{dtype} dims: [{size}]{optional} }}'
            )

        lines = [
            f'name: "{name}"',
            'backend: "python"',
            "max_batch_size: 0",
            "input [\n" + ",\n".join((tensor(*item) for item in inputs)) + "\n]",
            "output [\n" + ",\n".join((tensor(*item) for item in outputs)) + "\n]",
            (
                "instance_group [{ count: 1 kind: KIND_CPU }]"
                if provider == "CPUExecutionProvider"
                else "instance_group [{ count: 1 kind: KIND_GPU gpus: [0] }]"
            ),
        ]
        for key, value in dict(
            model_version=(
                exported_prefix if milliseconds is not None else model.cfg.model_name
            ),
            provider=provider,
            mode=mode,
            max_streams=str(max_streams),
        ).items():
            lines.append(
                f'parameters {{ key: "{key}" value {{ string_value: "{value}" }} }}'
            )
        (directory / "config.pbtxt").write_text("\n".join(lines) + "\n")
        print(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GigaAM for Triton")
    parser.add_argument("model_version")
    parser.add_argument(
        "--dynamic-context", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--contexts", type=int, nargs="*", default=[])
    parser.add_argument("--max-streams", type=int, default=32)
    parser.add_argument("--repository", default="repos")
    parser.add_argument("--provider", default="TensorRT-fp16", choices=BACKENDS)
    args = parser.parse_args()
    model_version = args.model_version
    model = gigaam.load_model(model_version, device="cpu")
    if not isinstance(model, gigaam.GigaAMDualASR):
        raise ValueError("Triton export requires a v4 dual-head model")
    convert_dual_repository(
        model,
        args.contexts,
        args.repository,
        args.provider,
        args.dynamic_context,
        args.max_streams,
    )


if __name__ == "__main__":
    main()
