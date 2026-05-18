import math
from typing import List, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ...base import VQAConfig, VQAModelBase


class InternVL3(VQAModelBase):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, config: VQAConfig):
        super().__init__(config)
        self.config = config
        self._load_model(config.model_path)

    def _load_model(self, model_path):
        # model_name = model_path.split("/")[-1]
        device_map = split_model(model_path, self.config.n_gpus)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self.config.data_type,
            # cache_dir="test",
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            device_map=device_map,
            cache_dir=self.config.save_cache_dir,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            cache_dir=self.config.save_cache_dir,
        )
        self.transform = build_transform(self.config.input_size)
        self.generation_config = dict(max_new_tokens=self.config.max_new_tokens, do_sample=False)

    def preprocess(self, image: Union[str, np.ndarray], prompts: Union[str, List[str]]):
        pass  # 사용하지 않음

    def video_preprocess(self, videos: List[np.ndarray], num_segments=None):
        """
        비디오 numpy 배열로부터 전처리
        videos: List of video arrays with shape (T, H, W, C) where T is number of frames
        num_segments: 샘플링할 프레임 수 (None이면 모든 프레임 사용)
        """
        assert isinstance(videos, list), "videos must be a list of video arrays"
        pixel_values_list = []
        num_patches_list = []
        transform = build_transform(self.config.input_size)

        for video in videos:
            assert isinstance(video, np.ndarray) and video.ndim == 4, \
                "Each video must be a 4D numpy array (T, H, W, C)"

            total_frames = video.shape[0]

            # 프레임 샘플링
            if num_segments is not None and num_segments < total_frames:
                frame_indices = np.linspace(0, total_frames - 1, num_segments, dtype=int)
            else:
                frame_indices = np.arange(total_frames)

            video_patches = []
            for frame_idx in frame_indices:
                frame = video[frame_idx]  # (H, W, C)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_img = Image.fromarray(frame)

                # dynamic_preprocess로 타일링 (비디오는 max_num=1 사용)
                tiles = dynamic_preprocess(
                    frame_img,
                    image_size=self.config.input_size,
                    use_thumbnail=True,
                    max_num=1,  # 비디오는 1로 고정
                )

                transformed = [transform(tile) for tile in tiles]
                tile_tensor = torch.stack(transformed).to(
                    dtype=self.config.data_type, device=self.config.device
                )

                video_patches.append(tile_tensor.size(0))
                pixel_values_list.append(tile_tensor)

            num_patches_list.append(video_patches)

        pixel_values = torch.cat(pixel_values_list, dim=0)
        return pixel_values, num_patches_list

    def image_preprocess(self, images: List[np.ndarray]):
        assert isinstance(images, list), "images must be a list of images"
        pixel_values_list = []
        transform = build_transform(self.config.input_size)
        index = []

        for image in images:
            assert isinstance(image, np.ndarray), "Each image must be a numpy array"

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            tiles = dynamic_preprocess(
                image,
                image_size=self.config.input_size,
                use_thumbnail=True,
                max_num=self.config.frame_per_tile_max_num,
            )
            transformed = [transform(tile) for tile in tiles]
            tile_tensor = torch.stack(transformed).to(
                dtype=self.config.data_type, device=self.config.device
            )
            index.append(tile_tensor.size(0))
            pixel_values_list.append(tile_tensor)

        pixel_values = torch.cat(pixel_values_list, dim=0)

        return pixel_values, index

    def forward(
        self, images: Union[str, List[np.ndarray]], prompts: Union[str, List[str]]
    ) -> List[str]:
        if isinstance(images, list):
            if len(images) > 0 and isinstance(images[0], np.ndarray):
                if images[0].ndim == 4:  # 비디오 데이터 (T, H, W, C)
                    pixel_values, num_patches_list = self.video_preprocess(videos=images , num_segments=self.config.num_segments)
                    return self.multi_batch_video_chat(pixel_values, num_patches_list, prompts)
                elif images[0].ndim == 3:  # 이미지 데이터 (H, W, C)
                    pixel_values, index = self.image_preprocess(images=images)
                    return self.multi_batch_chat(pixel_values, index, prompts)
        if isinstance(images, np.ndarray):
            if images.ndim == 4:  # 단일 비디오
                pixel_values, num_patches_list = self.video_preprocess(videos=[images] , num_segments=self.config.num_segments)
                return self.single_batch_video_chat(pixel_values, num_patches_list[0], prompts)
            elif images.ndim == 3:  # 단일 이미지
                pixel_values, index = self.image_preprocess(images=[images])
                return self.single_batch_chat(pixel_values, prompts)

    def single_batch_chat(self, pixel_values: np.ndarray, prompt: str) -> str:
        return self.model.chat(self.tokenizer, pixel_values, prompt, self.generation_config)

    def multi_batch_chat(self, pixel_values: List[np.ndarray], index: List[int], prompts: List[str]):

        return self.model.batch_chat(
            self.tokenizer,
            pixel_values,
            num_patches_list=index,
            questions=prompts,
            generation_config=self.generation_config,
        )

    def single_batch_video_chat(self, pixel_values: torch.Tensor, num_patches: List[int], prompt: str) -> str:
        """
        video 전용 prefix 생성 필수적임
        """
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches))])

        full_prompt = video_prefix + prompt

        return self.model.chat(
            self.tokenizer,
            pixel_values,
            full_prompt,
            self.generation_config,
            num_patches_list=num_patches
        )

    def multi_batch_video_chat(self, pixel_values: torch.Tensor, num_patches_list: List[List[int]], prompts: List[str]) -> List[str]:
        """
        비디오 배치 처리 (batch_chat 사용 가능!)
        num_patches_list: [[1,1,...,1], [1,1,...,1]] - 각 비디오의 프레임별 타일 수
        """
        full_prompts = []
        total_patches_per_video = []

        for num_patches, prompt in zip(num_patches_list, prompts):
            video_prefix = ''.join([f'Frame{j+1}: <image>\n' for j in range(len(num_patches))])
            full_prompts.append(video_prefix + prompt)
            total_patches_per_video.append(sum(num_patches))

        # print(f"total_patches_per_video: {total_patches_per_video}")
        # print(f"<image> counts: {[p.count('<image>') for p in full_prompts]}")

        return self.model.batch_chat(
            self.tokenizer,
            pixel_values,
            num_patches_list=total_patches_per_video,  # [v1, v2] - 각 비디오의 총 타일
            questions=full_prompts,                     # N개 질문
            generation_config=self.generation_config,
        )

#################################################################################


def build_transform(input_size):
    MEAN, STD = InternVL3.IMAGENET_MEAN, InternVL3.IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(np_image, input_size=448, max_num=12):
    image = Image.fromarray(cv2.cvtColor(np_image)).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_model(model_path, world_size):
    device_map = {}
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map
