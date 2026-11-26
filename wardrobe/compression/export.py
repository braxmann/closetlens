"""Make models smaller (M7): export a PyTorch model to ONNX, quantize it to int8, run it with ONNX Runtime.

ONNX is only a file format for the same model, like saving a Word document as PDF. We need it because the
int8 quantization tools of ONNX Runtime work on ONNX files, not on PyTorch models.
"""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static,
)
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
    # The exporter also writes a "shape note" (value_info) for every tensor, weights included. The quantizer
    # transposes weight matrices while rebuilding the graph and then contradicts those notes ("Inferred shape and
    # existing shape differ"). The notes are optional: ONNX Runtime works the shapes out itself, so drop them.
    graph = onnx.load(path)
    del graph.graph.value_info[:]
    onnx.save(graph, path)


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    """Dynamic int8 quantization: weights are rounded to int8 once, here; activations at runtime, per input.

    No calibration images needed. Meant for transformers, which are almost only MatMul/Gemm. It also rounds
    convolutions, but there it fails: on MobileNetV3 category fell from 0.95 to 0.13 (see quantize_int8_static).
    """
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,  # signed: -128..127, weights are spread around 0
        per_channel=True,  # one scale per output neuron instead of one per matrix: more precise
    )


class _ListReader(CalibrationDataReader):
    """Hands the calibration images to ONNX Runtime one by one (its interface: get_next, None when done)."""

    def __init__(self, inputs: list[np.ndarray]):
        self.batches = iter([{"image": x} for x in inputs])

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self.batches, None)


def quantize_int8_static(fp32_path: Path, int8_path: Path, calibration_inputs: list[np.ndarray]) -> None:
    """Static int8 quantization: weights *and* activations get fixed int8 scales, measured on calibration images.

    Dynamic quantization only covers MatMul/Gemm. A CNN like MobileNet is mostly convolutions, which only
    static quantization reaches. The calibration images run through the model once; the range of values seen
    at every layer decides its scale. They must be training images: choosing anything on the validation set
    would make the validation numbers too optimistic.
    """
    # Recommended preparation step: simplify the graph and recompute shapes. The symbolic part is skipped, it
    # crashes on the "Expand" node of SigLIP's attention pooling; our batch dimension is the only dynamic one anyway.
    prepared_path = fp32_path.with_suffix(".prepared.onnx")
    quant_pre_process(fp32_path, prepared_path, skip_symbolic_shape=True)
    quantize_static(
        model_input=prepared_path,
        model_output=int8_path,
        calibration_data_reader=_ListReader(calibration_inputs),
        quant_format=QuantFormat.QDQ,  # stores "quantize -> dequantize" pairs around each layer; the current format
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,  # activations after ReLU-like layers are mostly >= 0: unsigned fits
        # How the range of each layer is chosen. MinMax (the default) takes the largest value ever seen, so a
        # single outlier stretches the 256 steps until all normal values share a few of them: on MobileNetV3,
        # category fell from 0.95 to 0.63 (100 images) and to 0.14 (300 images: more chances for outliers).
        # Percentile ignores the most extreme 0.001 % and keeps the steps fine: 0.93.
        calibrate_method=CalibrationMethod.Percentile,
    )
    prepared_path.unlink()


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
