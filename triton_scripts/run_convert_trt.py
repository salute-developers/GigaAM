"""Convert ONNX graphs with trtexec and preserve graph I/O order."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import onnx


def conversion_arguments(path, target, max_batch, max_frames):
    graph = onnx.load(str(path), load_external_data=False).graph
    manifest_path = path.parent / "streaming.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    factor = manifest.get("subsampling_factor", 1)
    frames = (1, min(512, max_frames), max_frames)
    steps = (1, min(128, max_frames), max_frames)
    mel_max = max(factor, (max_frames + factor - 1) // factor * factor)
    times = dict(
        mel_frames=(factor, min(512, mel_max), mel_max),
        frames=frames,
        seq_len=frames,
        input_frames=frames,
        segment=(3, min(64, max(3, max_frames)), max(3, max_frames)),
        encoded=steps,
        step=steps,
    )
    profiles = [[], [], []]
    for value in graph.input:
        bounds = []
        for axis, dimension in enumerate(value.type.tensor_type.shape.dim):
            name = dimension.dim_param
            if axis == 2 and value.name in ("encoded", "segment"):
                name = value.name
            elif axis == 1 and name == "frames":
                name = "step"
            bounds.append(
                (dimension.dim_value,) * 3
                if dimension.HasField("dim_value")
                else times.get(name, (1, min(2, max_batch), max_batch))
            )
        for profile, shape in zip(profiles, zip(*bounds)):
            profile.append(value.name + ":" + "x".join(map(str, shape)))
    target.parent.mkdir(parents=True, exist_ok=True)

    def metadata(value):
        tensor = value.type.tensor_type
        return dict(
            name=value.name,
            dtype=onnx.TensorProto.DataType.Name(tensor.elem_type),
            shape=[
                dim.dim_value if dim.HasField("dim_value") else -1
                for dim in tensor.shape.dim
            ],
        )

    Path(str(target) + ".json").write_text(
        json.dumps(
            dict(
                max_batch=max_batch,
                inputs=list(map(metadata, graph.input)),
                outputs=list(map(metadata, graph.output)),
            )
        )
    )
    return [
        "--" + name + "Shapes=" + ",".join(shape)
        for name, shape in zip(("min", "opt", "max"), profiles)
    ]


def _default_max_frames(model):
    name = Path(model).name
    if name.endswith("_packet_decode.onnx"):
        return 1024
    if "v4_rnnt" in name and "stream_step" in name:
        return 2048
    if "v4_rnnt" in name and "stream_frontend" in name:
        return 8192
    return 4096


def _write_triton_child(source, plan):
    parent = source.parent.parent.parent
    config = parent / "config.pbtxt"
    if not config.exists() or not any(
        f'key: "provider" value {{ string_value: "{provider}"' in config.read_text()
        for provider in ("TensorRT", "TensorRT-fp16")
    ):
        return None
    name = f"{parent.name}_graph_{source.stem}"
    child = parent.parent / name
    version = child / "1"
    version.mkdir(parents=True, exist_ok=True)
    model_plan = version / "model.plan"
    target = os.path.relpath(plan, version)
    if model_plan.is_symlink() and os.readlink(model_plan) != target:
        model_plan.unlink()
    if not model_plan.exists():
        model_plan.symlink_to(target)
    metadata = json.loads(Path(str(plan) + ".json").read_text())
    types = {"FLOAT": "FP32", "FLOAT16": "FP16", "DOUBLE": "FP64"}

    def tensors(items):
        return ",\n".join(
            f'{{ name: "{item["name"]}" data_type: TYPE_{types.get(item["dtype"], item["dtype"])} '
            f"dims: [{', '.join(map(str, item['shape']))}] }}"
            for item in items
        )

    (child / "config.pbtxt").write_text(
        f'name: "{name}"\nplatform: "tensorrt_plan"\nmax_batch_size: 0\n'
        f"input [\n{tensors(metadata['inputs'])}\n]\n"
        f"output [\n{tensors(metadata['outputs'])}\n]\n"
        "instance_group [{ count: 1 kind: KIND_GPU gpus: [0] }]\n"
    )
    return name


def _engine_metadata(plan, layer_info):
    metadata = json.loads(Path(str(plan) + ".json").read_text())
    formats = {}
    for layer in json.loads(layer_info.read_text())["Layers"]:
        for tensor in [*layer.get("Inputs", []), *layer.get("Outputs", [])]:
            match = re.search(
                r"\b(FP32|FLOAT|FP16|HALF|FP64|INT32|INT64|BOOL)\b",
                tensor["Format/Datatype"],
                re.I,
            )
            if match:
                kind = match[0].upper()
                formats[tensor["Name"]] = {
                    "FP32": "FLOAT",
                    "FP16": "FLOAT16",
                    "HALF": "FLOAT16",
                    "FP64": "DOUBLE",
                }.get(kind, kind)
    for item in metadata["inputs"] + metadata["outputs"]:
        item["dtype"] = formats[item["name"]]
    Path(str(plan) + ".json").write_text(json.dumps(metadata))


def convert(model, fp16=None, max_batch=4, max_frames=None):
    source = Path(model)
    automatic = fp16 is None
    if automatic:
        config = source.parent.parent.parent / "config.pbtxt"
        fp16 = (
            not config.is_file() or 'string_value: "TensorRT"' not in config.read_text()
        )
    suffix = ".fp16.plan" if fp16 else ".plan"
    target = source.with_suffix(suffix)
    if max_frames is None:
        max_frames = _default_max_frames(source)
    profiles = conversion_arguments(source, target, max_batch, max_frames)
    layer_info = Path(str(target) + ".layers.json")
    command = [
        os.environ.get("TRTEXEC", "trtexec"),
        f"--onnx={source}",
        f"--saveEngine={target}",
        "--noTF32",
        "--memPoolSize=workspace:8192",
        "--profilingVerbosity=detailed",
        "--iterations=1",
        "--warmUp=0",
        "--duration=0",
        "--dumpLayerInfo",
        "--exportLayerInfo=" + str(layer_info),
        profiles[1].replace("--optShapes=", "--shapes="),
        *(["--fp16"] if fp16 else []),
        *profiles,
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        if not (automatic and fp16 and source.stem.endswith("_stream_step")):
            raise
        print(
            f"FP16 build failed for {source.name}. Retrying this graph in FP32.",
            flush=True,
        )
        command.remove("--fp16")
        subprocess.run(command, check=True)
    _engine_metadata(target, layer_info)
    child = _write_triton_child(source, target)
    if child:
        print(f"Created Triton TensorRT child model {child}", flush=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="v4 ONNX graph files")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Maximum time extent in graph input units; defaults to the largest "
            "validated profile for that graph"
        ),
    )
    args = parser.parse_args()
    for model in args.models:
        print(convert(model, args.fp16, args.max_batch, args.max_frames), flush=True)
