"""
script to run inference on tflite model on NPU
Usage:
python inference.py --model mobilenetv4_conv_large.e500_r256_in1k_full_integer_quant.tflite -r 320 --bhwc --npu
python inference.py --model ssd_mobilenet_v2_coco_quant_postprocess.tflite -r 300 --bhwc --image_path data/1.jpg --npu
python inference.py --model unet_320/unet_e12_lr0.0010_wd0.0100_r320_bce-loss_bilinear_width1.0_pw1.00_coco-person_full_integer_quant.tflite -r 320 --bhwc --image_path data/1.jpg --npu
python inference.py --model yolov8n_saved_model/yolov8n_full_integer_quant.tflite -r 320 --bhwc --image_path data/1.jpg --npu
python inference.py --model fpn/mobilenetv4_conv_medium_fpn_r320_full_integer_quant.tflite -r 320 --bhwc --image_path data/1.jpg --npu
"""

import numpy as np
import tflite_runtime.interpreter as tflite
import time
from pathlib import Path
import cv2
import argparse

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocess(img_path, w, h):
    img = cv2.imread(img_path)
    # resize
    img = cv2.resize(img, (w, h))
    # to 0-1
    img = img.astype(np.float32) / 255
    # to rgb
    img = img[:, :, ::-1]
    # normalize
    mean = np.array(MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(STD, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    # to nhwc
    img = img[np.newaxis, :]
    return img


def inference(args):
    if args.dtype == "int8":
        dtype = np.int8
    elif args.dtype == "uint8":
        dtype = np.uint8
    elif args.dtype == "float16":
        dtype = np.float16
    elif args.dtype == "float32":
        dtype = np.float32
    else:
        raise ValueError("Invalid dtype")

    tf_lite_model_path = Path(args.folder) / args.model

    if args.npu:
        delegate = tflite.load_delegate("/usr/lib/libvx_delegate.so")
        interpreter = tflite.Interpreter(
            model_path=str(tf_lite_model_path), experimental_delegates=[delegate]
        )

    else:
        interpreter = tflite.Interpreter(model_path=str(tf_lite_model_path))

    interpreter.allocate_tensors()
    # suppose only one input tensor, therefore [0]
    input_details = interpreter.get_input_details()
    # suppose one or multiple output tensors
    output_details = interpreter.get_output_details()

    print("input_details:")
    print(input_details)
    print("output_details:")
    print(output_details)

    input_shape = input_details[0]["shape"]
    input_width = input_shape[1]
    input_height = input_shape[2]

    ## preprocess: norm and resize
    input_shape = input_details[0]["shape"]
    input_width = input_shape[1]
    input_height = input_shape[2]
    if args.image_path:
        input = preprocess(args.image_path, input_width, input_height)
    else:  #  use dummy variable
        if args.bhwc:  # for Converter1
            input = np.random.randn(1, args.image_size, args.image_size, 3).astype(
                dtype
            )
        else:  # for Converter2
            input = np.random.randn(1, 3, args.image_size, args.image_size).astype(
                dtype
            )

    scale, zero_point = input_details[0]["quantization"]
    img_uint8 = (input / scale + zero_point).astype(dtype)
    if not args.bhwc:
        img_uint8 = np.transpose(img_uint8, (0, 3, 1, 2))
    # print(img_uint8.shape)
    interpreter.set_tensor(input_details[0]["index"], img_uint8)

    # Run inference
    interpreter.invoke()

    outputs = []
    for i, output_detail in enumerate(output_details):
        output = interpreter.get_tensor(output_detail["index"])
        scale, zero_point = output_details[i]["quantization"]
        if scale != 0:
            if zero_point != 0:
                # print(f"scale {scale}, zero_point {zero_point}")
                output = (output.astype(np.float32) - zero_point) * scale
        print(output.shape)
        outputs.append(output)

    print(outputs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--npu", action="store_true", help="Use NPU for inference")
    parser.add_argument(
        "--dtype",
        choices=["int8", "int8", "float16", "float32"],
        default="uint8",
        help="Data type to be used. Must be one of int8, float16, or float32.",
    )
    parser.add_argument(
        "--folder",
        "-p",
        type=str,
        default="/root/chen/tflite_models/",
        help="folder name",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        # default="best_model_full_integer_quant_uint8.tflite",
        default="pretrained_unet_e12_lr0.0010_wd0.0100_r320_bce-loss_convtranspose_width1.0_pw1.00_coco-person_full_integer_quant.tflite",
        help="Model name",
    )
    parser.add_argument(
        "--bhwc", action="store_true", help="Use bhwc format for input tensor"
    )
    parser.add_argument("--image_path", "-i", type=str, default=None, help="Image path")
    parser.add_argument("--image_size", "-r", type=int, default=224, help="Image size")

    args = parser.parse_args()

    inference(args)
