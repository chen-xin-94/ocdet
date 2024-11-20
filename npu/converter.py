"""
Usage:
    # instantiate either Converter by passing the required arguments
    converter = Converter(...)

    # convert the model to tflite (calib data automatically prepared) and save to "tflite_model_path"
    converter.convert()
"""

import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path


class Converter:
    """
    torch -> onnx -> tf -> tflite

    using onnx_tf
    """

    MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def __init__(
        self,
        torch_model,  # instance of the torch model to be converted
        image_size,
        torch_model_path,  # weights of the torch model
        onnx_model_path,
        tf_folder_path,
        tflite_model_path,
        calib_data_path,
        opset_version=13,  # higher version caused errors for vx delegate
        coco_dir="/mnt/ssd2/xin/data/coco",
        num_calib_img=10,
    ):
        self.torch_model = torch_model
        self.image_size = image_size
        self.torch_model_path = torch_model_path
        self.onnx_model_path = Path(onnx_model_path)
        self.tf_folder_path = Path(tf_folder_path)
        self.tflite_model_path = Path(tflite_model_path)
        self.calib_data_path = Path(calib_data_path)
        self.opset_version = opset_version
        self.coco_dir = Path(coco_dir)
        self.num_calib_img = num_calib_img

        self.onnx_model_path.parent.mkdir(parents=True, exist_ok=True)

    def convert(self):

        if self.torch_model_path:  # load model weights
            self.torch_model.load_state_dict(
                torch.load(self.torch_model_path, map_location=torch.device("cpu"))
            )
        torch_input = torch.randn(1, 3, *self.image_size)

        ## Step 1: Convert PyTorch to ONNX
        torch.onnx.export(
            self.torch_model.cpu(),
            torch_input.cpu(),
            self.onnx_model_path,
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        ## Step 2: Convert ONNX to TensorFlow
        onnx_model = onnx.load(self.onnx_model_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(self.tf_folder_path)

        ## Step 3: Convert TensorFlow to TFLite
        # Prepare calibration data
        if self.calib_data_path.exists():
            print("Using existing calibration data")
        else:
            print("Preparing calibration data")
            self.prepare_calibration_data()
        # Convert to tflite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(self.tf_folder_path))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.representative_data_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            # tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_model_quant = converter.convert()
        self.tflite_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.tflite_model_path.write_bytes(tflite_model_quant)

        # delete the calibration data
        self.calib_data_path.unlink()

    def prepare_calibration_data(self):
        files = (self.coco_dir / "val2017").glob("*.jpg")
        files = list(files)[: self.num_calib_img]  # TODO
        img_datas = []
        for _, file in enumerate(files):
            bgr_img = cv2.imread(str(file)).astype(np.float32)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(rgb_img, dsize=self.image_size)
            transpose_img = np.transpose(resized_img, (2, 0, 1))
            extend_batch_size_img = transpose_img[np.newaxis, :]
            normalized_img = extend_batch_size_img / 255.0
            img_datas.append(normalized_img)
        calib_data = np.vstack(img_datas)
        np.save(self.calib_data_path, arr=calib_data)

    def representative_data_gen(self):
        calib_data = np.load(self.calib_data_path)
        for idx in range(len(calib_data)):
            input_value = (calib_data[idx] - Converter.MEAN) / Converter.STD
            yield [input_value[np.newaxis, :]]
