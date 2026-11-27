"""BLS sessions for native TensorRT child models in Triton."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import triton_python_backend_utils as pb_utils


class TensorRTSession:
    def __init__(self, onnx_path, fp16=False):
        path = Path(onnx_path)
        parent = path.parents[2]
        self.name = f"{parent.name}_graph_{path.stem}"
        plan_suffix = ".fp16.plan.json" if fp16 else ".plan.json"
        self.metadata = json.loads(path.with_suffix(plan_suffix).read_text())
        self.max_batch = self.metadata.get("max_batch", 1)
        self.input_types = {
            item["name"]: self._types[item["dtype"]][0]
            for item in self.metadata["inputs"]
        }
        self.inputs, self.outputs = (
            [
                SimpleNamespace(name=item["name"], type=self._types[item["dtype"]][1])
                for item in self.metadata[key]
            ]
            for key in ("inputs", "outputs")
        )

    _types = {
        "FLOAT": (np.float32, "tensor(float)"),
        "FLOAT16": (np.float16, "tensor(float16)"),
        "DOUBLE": (np.float64, "tensor(double)"),
        "INT32": (np.int32, "tensor(int32)"),
        "INT64": (np.int64, "tensor(int64)"),
        "BOOL": (np.bool_, "tensor(bool)"),
    }

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, output_names, inputs):
        outputs = output_names or [item.name for item in self.outputs]
        request = pb_utils.InferenceRequest(
            model_name=self.name,
            requested_output_names=outputs,
            preferred_memory=pb_utils.PreferredMemory(
                pb_utils.TRITONSERVER_MEMORY_CPU, 0
            ),
            inputs=[
                pb_utils.Tensor(
                    node.name,
                    np.ascontiguousarray(
                        inputs[node.name], dtype=self.input_types[node.name]
                    ),
                )
                for node in self.inputs
            ],
        )
        response = request.exec()
        if response.has_error():
            raise RuntimeError(response.error().message())
        return [
            pb_utils.get_output_tensor_by_name(response, name).as_numpy()
            for name in outputs
        ]


class _LocalSession(TensorRTSession):
    def __init__(self, path, trt, device_id):
        super().__init__(path, fp16=True)
        self.cuda_device = torch.device("cuda", device_id)
        with torch.cuda.device(self.cuda_device):
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
            self.engine = self.runtime.deserialize_cuda_engine(
                Path(path).with_suffix(".fp16.plan").read_bytes()
            )
            if self.engine is None:
                raise RuntimeError(f"Cannot load the TensorRT plan for {path}")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError(f"Cannot create the TensorRT context for {path}")
        self.bindings = {}
        for entry in self.metadata["inputs"] + self.metadata["outputs"]:
            name = entry["name"]
            dtype = torch.from_numpy(np.empty((), self._types[entry["dtype"]][0])).dtype
            device = (
                self.cuda_device
                if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE
                else torch.device("cpu")
            )
            self.bindings[name] = dtype, device

    def run(self, output_names, inputs, keep_after=None):
        context = self.context
        keepalive = []
        for node in self.inputs:
            name = node.name
            dtype, device = self.bindings[name]
            value = (
                torch.as_tensor(inputs[name])
                .to(device=device, dtype=dtype, non_blocking=True)
                .contiguous()
            )
            keepalive.append(value)
            if not context.set_input_shape(name, tuple(value.shape)):
                raise ValueError(
                    f"TensorRT input {name} has unsupported shape {tuple(value.shape)}"
                )
            context.set_tensor_address(name, value.data_ptr())
        if missing := context.infer_shapes():
            raise RuntimeError(f"Cannot resolve TensorRT shapes for {missing}")
        outputs = {}
        for node in self.outputs:
            dtype, device = self.bindings[node.name]
            value = torch.empty(
                tuple(context.get_tensor_shape(node.name)), dtype=dtype, device=device
            )
            context.set_tensor_address(node.name, value.data_ptr())
            outputs[node.name] = value
        stream = torch.cuda.current_stream(self.cuda_device)
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        selected = [outputs[name] for name in (output_names or outputs)]
        host = [
            (
                torch.empty_like(value, device="cpu", pin_memory=True).copy_(
                    value, non_blocking=True
                )
                if value.is_cuda
                else value
            )
            for value in selected[:keep_after]
        ]
        stream.synchronize()
        return [*(value.numpy() for value in host), *selected[len(host) :]]

    def run_cached(self, values, caches):
        inputs = dict(zip((node.name for node in self.inputs), values))
        nodes = self.inputs[len(values) :]
        first = caches[0]
        if isinstance(first, tuple) and all(
            isinstance(row, tuple) and row[0] is first[0] and row[1] == first[1] + i
            for i, row in enumerate(caches)
        ):
            start = first[1]
            packed = [value[start : start + len(caches)] for value in first[0]]
        else:
            rows = [
                (
                    [value[cache[1] : cache[1] + 1] for value in cache[0]]
                    if isinstance(cache, tuple)
                    else cache
                )
                for cache in caches
            ]
            packed = [
                torch.cat(
                    [
                        torch.as_tensor(value, device=self.bindings[node.name][1])
                        for value in parts
                    ]
                )
                for node, parts in zip(nodes, zip(*rows), strict=True)
            ]
        inputs.update(
            (node.name, value) for node, value in zip(nodes, packed, strict=True)
        )
        return self.run(None, inputs, keep_after=4)


def session_factory(provider, device_id=0, direct_streaming=False):
    if provider not in ("TensorRT", "TensorRT-fp16"):
        raise ValueError(f"Unsupported native TensorRT provider: {provider}")

    def create(path):
        fp16 = provider == "TensorRT-fp16"
        name = Path(path).stem
        local = (
            name.endswith("_encoder")
            or direct_streaming
            and name.endswith(("_stream_step", "_stream_frontend", "_packet_decode"))
        )
        if fp16 and local:
            try:
                import tensorrt_bindings as trt
            except ImportError:
                pass
            else:
                return _LocalSession(path, trt, device_id)
        return TensorRTSession(path, fp16=fp16)

    return create
