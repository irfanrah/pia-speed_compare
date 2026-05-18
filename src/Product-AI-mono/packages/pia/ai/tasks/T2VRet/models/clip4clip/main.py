import os.path
from typing import List, Literal, Tuple

import numpy as np
import torch
from numpy import ndarray
from pia.utils.exception.model_handler import raise_exception_decorator
from pia.utils.exception.load_model import load_state_dict_with_mismatch
from pia.vision.preprocessing import video_preprocess
from torchvision.transforms import (
    Compose,
    Normalize,
    ToPILImage,
    ToTensor,
)

from ...base import T2VRetConfig, T2VRetModelBase, T2VRetONNXConfig
from .model import CLIP4Clip
from .tokenization_clip import SimpleTokenizer


class Clip4Clip(T2VRetModelBase):
    SPECIAL_TOKEN = {
        "CLS_TOKEN": "<|startoftext|>",
        "SEP_TOKEN": "<|endoftext|>",
        "MASK_TOKEN": "[MASK]",
        "UNK_TOKEN": "[UNK]",
        "PAD_TOKEN": "[PAD]",
    }

    def __init__(self, config: T2VRetConfig):
        self.config = config
        self.model = self.__load_model__()
        self.transform = Compose(
            [
                ToPILImage(),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.non_normalized_transform = Compose(
            [
                ToPILImage(),
                ToTensor(),
                # onnx export의 경우 Normalize는 모델 레이어 안으로 삽입함
            ]
        )
        self.tokenizer = SimpleTokenizer()

    @raise_exception_decorator(FileNotFoundError)
    def _load_model(self):
        state_dict = torch.load(self.config.model_path, map_location=self.config.device)
        model = CLIP4Clip(clip_state_dict=state_dict, task_cfg=self.config)
        load_state_dict_with_mismatch(model, state_dict)
        if self.config.use_half_precision:
            model.eval().half()
        else:
            model.eval().float()
        model.to(device=self.config.device)
        return model

    def video_preprocess(
        self, video: np.ndarray, device: Literal["cuda", "cpu", "mps"] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess the video

        Args:
            video (np.ndarray): Video array. Shape: (Batch size, Sequence length, Height, Width, Channel).
            device (str, optional): Device. "cuda" or "cpu" or "mps". Defaults to None.

        Returns:
            video_tensor (torch.Tensor): Preprocessed video tensor. Shape: (Batch size, Number of tiles, Sequence length, Channel, Height, Width).
            video_mask_tensor (torch.Tensor): Video mask tensor. Shape: (Batch size, Number of tiles, Sequence length).
        """

        # TODO : numpy, tensor 둘다 가능한지 확인 필요
        video_tensor, video_mask_tensor = video_preprocess(
            video=video,
            device=self.config.device if not device else device,
            tile_size=self.config.tile_config,
            img_size=self.config.img_size,
            transform=self.transform,
        )

        return video_tensor, video_mask_tensor

    def text_preprocess(self, texts: str | List[str]):
        input_ids_list = []
        input_mask_list = []
        if isinstance(texts, str):
            texts = [texts]
        for text in texts:
            text = self.tokenizer.tokenize(text)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + text
            total_length_with_CLS = self.config.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)

            # padding
            input_mask = [1] * len(input_ids) + [0] * (self.config.max_words - len(input_ids))
            input_ids = input_ids + [0] * (
                self.config.max_words - len(input_ids)
            )  # padding을 0으로 채웠나?
            assert len(input_ids) == self.config.max_words

            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
        return (
            torch.tensor(input_ids_list, device=self.config.device),
            torch.tensor(input_mask_list, device=self.config.device),
        )

    def preprocess(self, video: torch.Tensor | ndarray, text: str | List[str]):
        video, video_mask = self.video_preprocess(video=video)
        txt_ids, txt_mask = self.text_preprocess(texts=text)
        return video, video_mask, txt_ids, txt_mask

    def encode_text(
        self,
        text: torch.Tensor | ndarray | str,
        txt_mask=None,
    ):
        if isinstance(text, str) or isinstance(text, list):
            txt_ids, txt_mask = self.text_preprocess(texts=text)
        return self.model.get_sequence_output(
            input_ids=txt_ids,
            token_type_ids=txt_ids,
            attention_mask=txt_mask,
        )

    def encode_video_with_numpy(self, video: ndarray, video_mask=None):
        """
        Encode video to embedding vector.

        The `self.video_preprocess()` method preprocesses the input `video`
        numpy array, converting its data type to a torch.Tensor. This
        preprocessing also splits the frames into tiles, changing the shape from
        (Batch size, Sequence length, Height, Width, Channel) to (Batch size,
        Number of tiles, Sequence length, Height, Width, Channel).

        The `self.model.get_visual_output()` method then encodes the
        preprocessed `video` torch.Tensor into a video embedding vector. This
        results in `visual_vector` which has the shape (Batch size, Number of
        tiles, Embedding dimension).

        Args:
            video (ndarray): Video numpy array. Shape: (Batch size, Sequence
                length, Height, Width, Channel)
            video_mask (ndarray, optional): Video mask. Defaults to None.

        Returns:
            torch.Tensor: Video embedding vector. Shape: (Batch size, Number of
                tiles, Embedding dimension)

        Raises:
            AssertionError: If the input `video` is not of type `ndarray`.
        """
        assert (
            isinstance(video, ndarray)
        ), "Only ndarray is allowed. Tensor is not allowed. Tensor input skips video preprocessing such as normalization, resize, etc."

        video, video_mask = self.video_preprocess(video=video)
        if video_mask is None:
            batch_size, num_tiles, sequence_length, height, width, channel = video.shape
            video_mask = torch.ones(
                (batch_size, num_tiles, sequence_length),
                dtype=torch.int64,
                device=self.config.device,
            )

        visual_vector = self.model.get_visual_output(video=video, video_mask=video_mask)
        return visual_vector

    def encode_video_with_torch(self, video: torch.Tensor, video_mask=None):
        assert (
            isinstance(video, torch.Tensor)
        ), "Only torch.Tensor is allowed. Tensor is not allowed. Tensor input skips video preprocessing such as normalization, resize, etc."

        if video_mask is None:
            batch_size, num_tiles, sequence_length, height, width, channel = video.shape
            video_mask = torch.ones(
                (batch_size, num_tiles, sequence_length),
                dtype=torch.int64,
                device=self.config.device,
            )

        visual_vector = self.model.get_visual_output(video=video, video_mask=video_mask)
        return visual_vector

    @torch.no_grad
    def forward(
        self,
        video: torch.Tensor | ndarray | None = None,
        video_mask=None,
        text: torch.Tensor | ndarray | str | None = None,
        txt_mask=None,
    ) -> ndarray:
        if video is None:  # only text
            return self.encode_text(text=text, txt_mask=txt_mask)
        if isinstance(video, list):
            video = np.array(video)

        elif text is None:  # only video
            if isinstance(video, ndarray):
                return self.encode_video_with_numpy(video=video)
            elif isinstance(video, torch.Tensor):
                return self.encode_video_with_torch(video=video)
            else:
                raise "Only torch.Tensor or numpy ndarray is allowed"

        if txt_mask is None:
            txt_ids, txt_mask = self.text_preprocess(texts=text)

        if video_mask is None and isinstance(video, ndarray):
            video, video_mask = self.video_preprocess(video=video)
        elif video_mask is None and isinstance(video, torch.Tensor):
            B, F, _, _, _ = video.shape
            video_mask = torch.ones((B, F), device=self.config.device)

        txtual_vectors, visual_vectors = self.model.get_sequence_visual_output(
            input_ids=txt_ids,
            token_type_ids=txt_ids,  # 왜 있는지 모르겠음
            attention_mask=txt_mask,
            video=video,
            video_mask=video_mask,
        )
        # Calculate similarity scores
        #   similarity.shape = (Number of captions, Number of video batches)
        similarity = self.model._loose_similarity(
            sequence_output=txtual_vectors,
            visual_output=visual_vectors,
        )
        return similarity.cpu().detach().numpy()

    def fixed_text_vector_forward(self, video, txtual_vectors):
        B, F, _, _, _ = video.shape
        video_mask = torch.ones((B, F), device=self.config.device)
        visual_vectors = self.model.get_visual_output(video, video_mask)

        return self.model._loose_similarity(
            sequence_output=txtual_vectors, visual_output=visual_vectors
        )

    def export(self, onnx_config: T2VRetONNXConfig):
        dtypes = torch.float16 if onnx_config.half else torch.float32
        if onnx_config.split_part == "visual":
            new_model = VisualModel(self.model)
            input_size = {
                "input0": [
                    1,
                    3,
                    self.config.img_size[0],
                    self.config.img_size[1],
                ]
            }
            input_dummy = torch.rand(input_size["input0"], dtype=dtypes).to(self.config.device)

        elif onnx_config.split_part == "textual":
            new_model = TextualModel(self.model)
            input_size = {
                "input0": [
                    1,
                    self.config.max_words,
                    self.model.clip.token_embedding.embedding_dim,
                ],
                "input1": [1, self.config.max_words],
            }
            input_dummy = (
                torch.rand(input_size["input0"], dtype=dtypes).to(self.config.device),
                torch.rand(input_size["input1"], dtype=dtypes).to(self.config.device),
            )

        elif onnx_config.split_part == "embedding":
            new_model = self.model.clip.token_embedding
            input_size = {
                "input0": [1, self.config.max_words],
            }
            input_dummy = torch.randint(
                0,
                self.model.clip.vocab_size,
                input_size["input0"],
                dtype=torch.int32,
            ).to(
                self.config.device
            )  # embedding = only int

        onnx_config.input_size = input_size
        onnx_config.input_dummy = input_dummy
        self.export2onnx(onnx_config, new_model)

    def model2pt(self, output_pt_path: str):
        """
        Save the model to a .pt file.

        Args:
            output_pt_path (str): Output path for the .pt file.

        Returns:
            None
        """
        if not os.path.exists(os.path.basename(output_pt_path)):
            os.makedirs(os.path.basename(output_pt_path), exist_ok=True)
        torch.save(self.model, output_pt_path)
        print(f"Model saved to {output_pt_path}")


class VisualModel(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        # TODO : to("cuda") -> device 읽어와서 하도록 수정 필요
        self.mean = (
            torch.tensor([0.48145466, 0.4578275, 0.40821073])
            .view(1, 3, 1, 1)
            .expand(1, 3, 224, 224)
            .to("cuda")
        )  # normalize layer가 필요없을시 주석처리
        self.std = (
            torch.tensor([0.26862954, 0.26130258, 0.27577711])
            .view(1, 3, 1, 1)
            .expand(1, 3, 224, 224)
            .to("cuda")
        )  # normalize layer가 필요없을시 주석처리
        self.hidden_model = model.clip.visual

    def forward(self, x):
        # hidden = self.visual(image.type(self.dtype), video_frame=video_frame)
        x = (x - self.mean) / self.std  # 전처리에 있던 Normalize를 모델로 넣음. normalize layer가 필요없을시 주석처리
        x = self.hidden_model(x)
        x = self.hidden_model.ln_post(x) @ self.hidden_model.proj
        x = x[:, 0, :]
        return x


class TextualModel(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.positional_embedding = model.clip.positional_embedding
        self.transformer = model.clip.transformer
        self.projection_layer = model.clip.text_projection
        self.ln_final = model.clip.ln_final

    def forward(self, x, mask):
        # x, mask = d
        pos_emb = self.positional_embedding[: x.size(1), :]
        x = x + pos_emb
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x) @ self.projection_layer
        x = x[torch.arange(x.shape[0]), mask.argmin(dim=-1) - 1]
        return x
