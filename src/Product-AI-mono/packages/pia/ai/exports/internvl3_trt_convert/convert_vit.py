"""
InternVL Vision Encoder (ViT) -> ONNX -> TensorRT engine converter.

Usage as module:
    from pia.ai.exports.vqa_trt_convert.convert_vit import build_vit_engine

    build_vit_engine(
        pretrained_model_path="/assets/InternVL3-2B",
        onnx_file="/assets/InternVL3-2B/vision_encoder_bfp16.onnx",
        trt_file="/assets/InternVL3-2B/vision_encoder_bfp16.trt",
        image_path="./data/frames/frame_0.jpg",
        dtype="bfloat16",
        min_bs=1,
        opt_bs=13,
        max_bs=26,
    )

Usage as CLI:
    python -m pia.ai.exports.vqa_trt_convert.convert_vit \
        --pretrainedModelPath /assets/InternVL3-2B \
        --onnxFile /assets/InternVL3-2B/vision_encoder_bfp16.onnx \
        --trtFile /assets/InternVL3-2B/vision_encoder_bfp16.trt \
        --imagePath ./data/frames/frame_0.jpg \
        --dtype bfloat16 --minBS 1 --optBS 13 --maxBS 26
"""

from __future__ import annotations

import argparse
import os
from time import time

import tensorrt as trt
import torch
import torchvision.transforms as T
from PIL import Image
from tensorrt_llm._utils import str_dtype_to_torch
from torchvision.transforms import InterpolationMode
from transformers import AutoModelForCausalLM

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------
def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448,
                       use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file: str, input_size: int = 448,
               max_num: int = 12) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size,
                                use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    print(f"pixel_values shape: {pixel_values.shape}")
    return pixel_values


# ---------------------------------------------------------------------------
# Vision encoder wrapper (HF model -> single forward)
# ---------------------------------------------------------------------------
class VisionEncoderWrapper(torch.nn.Module):

    def __init__(self, vision_model, vision_mlp1, select_layer=-1):
        super().__init__()
        self.vision_model = vision_model
        self.mlp1 = vision_mlp1
        self.downsample_ratio = 0.5
        self.select_layer = select_layer

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                    int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds,
                                        scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1,
                                        vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds


# ---------------------------------------------------------------------------
# ONNX export + TensorRT build helpers
# ---------------------------------------------------------------------------
def export_onnx(vision_encoder: torch.nn.Module, image: torch.Tensor,
                onnx_file: str) -> None:
    print("Start converting ONNX model!")
    torch.onnx.export(
        vision_encoder,
        args=image,
        f=onnx_file,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}},
    )
    print(f"Succeeded converting ONNX model: {onnx_file}")


def generate_trt_engine(onnx_file: str, trt_file: str, image_size: int = 448,
                        min_bs: int = 1, opt_bs: int = 13,
                        max_bs: int = 52) -> None:
    print("Start converting TRT engine!")
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.BF16)

    parser = trt.OnnxParser(network, logger)
    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read(), "/".join(onnx_file.split("/"))):
            print(f"Failed parsing {onnx_file}")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError(f"ONNX parse failed: {onnx_file}")
        print(f"Succeeded parsing {onnx_file}")

    inp = network.get_input(0)
    inp.shape = [-1, 3, image_size, image_size]
    profile.set_shape(
        inp.name,
        [min_bs, 3, image_size, image_size],
        [opt_bs, 3, image_size, image_size],
        [max_bs, 3, image_size, image_size],
    )
    config.add_optimization_profile(profile)

    t0 = time()
    engine_str = builder.build_serialized_network(network, config)
    t1 = time()
    if engine_str is None:
        raise RuntimeError(f"Failed building TRT engine: {trt_file}")
    print(f"Succeeded building {trt_file} in {t1 - t0:.0f} s")

    with open(trt_file, "wb") as f:
        f.write(engine_str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_vit_engine(
    pretrained_model_path: str,
    onnx_file: str,
    trt_file: str,
    image_path: str,
    dtype: str = "bfloat16",
    load_on_cpu: bool = False,
    only_trt: bool = False,
    min_bs: int = 1,
    opt_bs: int = 13,
    max_bs: int = 26,
    image_size: int = 448,
    max_num: int = 12,
) -> str:
    """Build a TensorRT engine for the InternVL vision encoder.

    Steps:
        1. Load the HuggingFace model
        2. Export vision encoder to ONNX (unless only_trt=True)
        3. Build TensorRT engine from ONNX

    Returns the trt_file path on success.
    """
    # ensure output dirs exist
    for path in (onnx_file, trt_file):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    # load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_path,
        device_map="cpu",
        torch_dtype=str_dtype_to_torch(dtype),
        trust_remote_code=True,
        use_flash_attn=False,
    ).eval()
    vision_encoder = VisionEncoderWrapper(
        hf_model.vision_model, hf_model.mlp1, select_layer=-1)

    # load sample image for tracing
    image = load_image(image_path, input_size=image_size,
                       max_num=max_num).to(str_dtype_to_torch(dtype))
    if not load_on_cpu:
        vision_encoder = vision_encoder.to("cuda")
        image = image.to("cuda")

    # export + build
    if only_trt:
        generate_trt_engine(onnx_file, trt_file, image_size=image_size,
                            min_bs=min_bs, opt_bs=opt_bs, max_bs=max_bs)
    else:
        export_onnx(vision_encoder, image, onnx_file)
        generate_trt_engine(onnx_file, trt_file, image_size=image_size,
                            min_bs=min_bs, opt_bs=opt_bs, max_bs=max_bs)

    print(f"ViT engine saved: {trt_file}")
    return trt_file


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TRT engine for InternVL vision encoder.",
    )
    parser.add_argument("--pretrainedModelPath", type=str, required=True)
    parser.add_argument("--onnxFile", type=str, required=True)
    parser.add_argument("--trtFile", type=str, required=True)
    parser.add_argument("--imagePath", type=str, default="./data/image1.jpg")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--loadOnCpu", action="store_true")
    parser.add_argument("--onlyTrt", action="store_true")
    parser.add_argument("--minBS", type=int, default=1)
    parser.add_argument("--optBS", type=int, default=13)
    parser.add_argument("--maxBS", type=int, default=26)
    return parser.parse_args()


def main():
    args = _parse_arguments()
    build_vit_engine(
        pretrained_model_path=args.pretrainedModelPath,
        onnx_file=args.onnxFile,
        trt_file=args.trtFile,
        image_path=args.imagePath,
        dtype=args.dtype,
        load_on_cpu=args.loadOnCpu,
        only_trt=args.onlyTrt,
        min_bs=args.minBS,
        opt_bs=args.optBS,
        max_bs=args.maxBS,
    )


if __name__ == "__main__":
    main()
