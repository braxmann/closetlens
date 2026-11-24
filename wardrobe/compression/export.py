"""Make models smaller (M7): export a PyTorch model to ONNX, quantize it to int8, run it with ONNX Runtime.

ONNX is only a file format for the same model, like saving a Word document as PDF. We need it because the
int8 quantization tools of ONNX Runtime work on ONNX files, not on PyTorch models.
"""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch import nn


def export_onnx(model: nn.Module, example_input: torch.Tensor, path: Path, output_names: list[str]) -> None:
    """Write the model as an ONNX file, with a variable batch size.

    The exporter runs the model once on example_input and records every operation on the way: that recording
    is the ONNX graph. The input therefore only needs the right shape and dtype, its content does not matter.
    A model that returns a dict gets one ONNX output per entry, in the dict's order: output_names names them.
    """
    # Without this the batch size of example_input would be baked into the graph.
    # Use an example with 2 images: torch.export treats a size of 1 as a constant, even when marked as dynamic.
    batch = torch.export.Dim("batch")
    torch.onnx.export(
        model.eval(),
        (example_input,),  # the arguments of model(...), always as a tuple
        path,
        input_names=["image"],  # the name under which ONNX Runtime expects the input later
        output_names=output_names,
        dynamic_shapes=({0: batch},),  # one entry per argument: dimension 0 of the first argument is the batch
        dynamo=True,  # the current exporter (based on torch.export); needs the onnxscript package
        # Default True stores the weights in a second file next to it (path + ".data"); model_size_mb would
        # then measure only the graph. Everything fits into one file up to 2 GB.
        external_data=False,
    )


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    """Dynamic int8 quantization: weights are rounded to int8 once, here; activations at runtime, per input.

    No calibration images needed. Only MatMul/Gemm weights are quantized: in a ViT those are almost all of them.
    """
    # Recommended preparation step of ONNX Runtime: recompute all tensor shapes and simplify the graph.
    # Without it, quantizing our heads fails: the exporter stores shape notes for the Gemm weights that
    # the quantizer's own shape check contradicts ("Inferred shape and existing shape differ").
    prepared_path = fp32_path.with_suffix(".prepared.onnx")
    quant_pre_process(fp32_path, prepared_path)
    quantize_dynamic(
        model_input=prepared_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,  # signed: -128..127, weights are spread around 0
        per_channel=True,  # one scale per output neuron instead of one per matrix: more precise
    )
    prepared_path.unlink()  # only an intermediate step


def onnx_predict(path: Path, providers: list[str] | None = None) -> Callable[[torch.Tensor], dict[str, np.ndarray]]:
    """Load an ONNX file and return a predict function with the same shape as the PyTorch ones.

    ONNX Runtime knows nothing about torch: it takes and returns numpy arrays.
    The providers decide where it runs; CPUExecutionProvider works everywhere.
    """
    # Loading is slow (reading the file, optimizing the graph), so it happens once, outside predict.
    session = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
    names = [output.name for output in session.get_outputs()]

    def predict(x: torch.Tensor) -> dict[str, np.ndarray]:
        # run(None, ...): None means "all outputs"; the result is a list with one array per output, in graph order.
        return dict(zip(names, session.run(None, {"image": x.numpy()})))

    return predict
