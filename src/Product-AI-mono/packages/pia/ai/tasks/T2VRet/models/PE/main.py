from typing import List, Literal, Tuple, Union

import numpy as np
import pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder.pe as pe
import pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder.transforms as transforms
import torch
from numpy import ndarray
from pia.utils.exception.model_handler import raise_exception_decorator
from pia.vision.preprocessing import PE_video_preprocess
from pia.ai.tasks.T2VRet.models.PE.utils.PE_class import PEModelInitializer
from pia.ai.tasks.T2VRet.models.PE.PE_config.main_cfg import *
from PIL import Image

from ...base import T2VRetConfig, T2VRetModelBase, T2VRetONNXConfig


class PerceptionEncoder(T2VRetModelBase):
    def __init__(self, config: T2VRetConfig):
        self.config = config
        self.PE_custom_config = PE_CUSTOM_CONFIG[self.config.model_name]
        self.model, self.transform, self.tokenizer = self._load_model()

    @raise_exception_decorator(FileNotFoundError)
    def _load_model(self):
        
        if self.config.model_path is not None:
            if self.config.model_path.endswith(".onnx"):
                raise NotImplementedError("ONNX model is not supported")

        # Determine if pretrained weights should be loaded
        is_pretrained = self.config.model_path is None

        if not is_pretrained:
            weight_path = self.config.model_path
            adapter_path = self.config.adapter_path
        else:
            weight_path = self.PE_custom_config.weight_path
            adapter_path = self.PE_custom_config.lora_adapter_path
        pe_init = PEModelInitializer(
                    model_name=self.PE_custom_config.original_model_name,
                    device=self.config.device,
                    pretrained=is_pretrained,
                    load_type=self.PE_custom_config.load_type,
                    weight_path=weight_path,
                    lora_adapter_path=adapter_path,
                    split_qkv=self.PE_custom_config.split_qkv,
                    pretrained_split_qkv_path=self.PE_custom_config.pretrained_split_qkv_path,
                    head_args=self.PE_custom_config.additional_head_args
                )
        model, preprocess, tokenizer, _, _ = pe_init.initialize()

        model.eval()
        model = model.half() if self.config.use_half_precision else model.float()

        return model, preprocess, tokenizer

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
        if self.config.tile_config is not None:
            raise NotImplementedError

        if all(size is not None for size in self.config.img_size):
            raise NotImplementedError

        video_tensor, video_mask_tensor = PE_video_preprocess(
            video=video,
            device=self.config.device if not device else device,
            tile_size=self.config.tile_config,
            img_size=self.config.img_size,
            transform=self.transform,
        )
        return video_tensor, video_mask_tensor

    def _preprocess_text(self, text: Union[str, List[str], torch.Tensor]):        
        if isinstance(text, str):
            return [text]
        elif isinstance(text, (list, torch.Tensor)):
            return text
        else:
            raise NotImplementedError(f"Unsupported input type: {type(text)}")

    def _matmul(
        self,
        model: torch.nn.Module,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        normalize: bool = False,
        logit_scale: bool = True,
    ):
        if normalize:
            embedding1 = embedding1 / embedding1.norm(dim=-1, keepdim=True)
            embedding2 = embedding2 / embedding2.norm(dim=-1, keepdim=True)

        if logit_scale:
            logit_scale = model.logit_scale.exp()
        else:
            logit_scale = 1
        sim_matrix = logit_scale * torch.matmul(embedding1, embedding2.t())
        return sim_matrix

    def preprocess(self, video: Union[np.ndarray, torch.Tensor], text: str | List[str]):
        video_tensor, _ = self.video_preprocess(video=video)
        tokens = self.tokenizer(self._preprocess_text(text))
        return video_tensor, tokens

    def encode_text(self, text: Union[np.ndarray, torch.Tensor] | str, normalize: bool = False):
        tokens = self.tokenizer(self._preprocess_text(text)).to(self.config.device)
        return self._encode_token(tokens, normalize)

    def _encode_token(self, tokens: torch.Tensor, normalize: bool = False):
        text_feat = self.model.encode_text(tokens)
        if normalize:
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
        return text_feat

    def encode_video_with_list(self,
                               video: List[np.ndarray],
                               video_mask=None,
                               normalize: bool = True
                               ):
        if self.config.tile_config is not None:
            raise NotImplementedError
        assert isinstance(video, list), "Only list is allowed"
        assert all(isinstance(v, np.ndarray) for v in video), "only np.ndarrat element is allowed "
        assert len(video[0].shape) == 3, "Only 3 dim frame is allowed"
        # video -> B, H, W, C
        preprocess_video = torch.stack([self.transform(Image.fromarray(v)) for v in video])
        preprocess_video = preprocess_video.to(self.config.device)
        preprocess_video = preprocess_video.unsqueeze(1)  # B, H, W, C -> B, 1, H, W, C, just analyze 1 image not having time squence. 시퀀스 축 없이 하나만 분석합니다.
        # video -> B, C, H, W
        video_feat = self.encode_video_with_torch(video=preprocess_video, normalize=normalize)
        return video_feat

    def encode_video_with_numpy(self, video: ndarray, video_mask=None, normalize: bool = False):
        """ """
        assert isinstance(video, ndarray), "Only ndarray is allowed."

        video_tensor, _ = self.video_preprocess(video=video)
        video_feat = self.model.encode_video(video_tensor)
        if normalize:
            video_feat /= video_feat.norm(dim=-1, keepdim=True)
        return video_feat

    def encode_video_with_torch(
        self, video: torch.Tensor, video_mask=None, normalize: bool = False
    ) -> torch.Tensor:
        """
        Encodes video input using the model's video encoder.

        Args:
            video (torch.Tensor): Input tensor of shape (T, H, W, C) or (B, T, H, W, C).
            video_mask (optional): Not used in current implementation.
            normalize (bool): Whether to normalize the output feature vector(s).

        Returns:
            torch.Tensor: Encoded video features of shape (B, D).
        """
        assert isinstance(video, torch.Tensor), "Input must be a torch.Tensor"

        if video.ndim == 4:
            # Single video: (T, H, W, C) → add batch dimension
            video = video.unsqueeze(0)
        elif video.ndim != 5:
            raise ValueError("Input tensor must be 4D or 5D")

        # Encode once
        video_feat = self.model.encode_video(video)

        # Optional normalization
        if normalize:
            video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        return video_feat

    @torch.no_grad
    def forward(
        self,
        video: torch.Tensor | ndarray | None = None,
        text: torch.Tensor | ndarray | str | None = None,
    ) -> np.ndarray:
        if video is None:  # only text
            return self.encode_text(text)

        if text is None:  # only video
            if isinstance(video, list):
                return self.encode_video_with_list(video=video)
            if isinstance(video, np.ndarray):
                video_tensor, _ = self.video_preprocess(video=video)
                video_feat = self.encode_video_with_torch(video=video_tensor, normalize=False)
                return video_feat
            elif isinstance(video, torch.Tensor):
                raise NotImplementedError
            else:
                raise TypeError("Only torch.Tensor or numpy ndarray is allowed")

        if isinstance(video, list):
            video = np.array(video)

        if isinstance(video, np.ndarray):
            video_tensor, _ = self.video_preprocess(video=video)
        elif isinstance(video, torch.Tensor):
            raise NotImplementedError

        video_feat = self.encode_video_with_torch(video=video_tensor, normalize=False)
        text_feat = self.encode_text(text, normalize=False)
        similarity_matrix = self._matmul(
            model=self.model,
            embedding1=text_feat,
            embedding2=video_feat,
            normalize=True,
            logit_scale=False,
        )
        similarity_matrix = similarity_matrix.cpu().detach().numpy()
        return similarity_matrix

    def export(self, onnx_config: T2VRetONNXConfig):
        raise NotImplementedError

    def model2pt(self, output_pt_path: str):
        raise NotImplementedError


if __name__ == "__main__":
    device = "cuda"
    model_name = "PE-Core-L14-336"
    model = pe.CLIP.from_config(model_name, pretrained=True).to(device)
    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
