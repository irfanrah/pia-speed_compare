import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List
from pia.ai.tasks.T2VRet.base import T2VRetConfig, T2VRetModelBase, T2VRetONNXConfig
from pia.ai.tasks.T2VRet.models.qwen3_vl_embedding.models.qwen3_vl_embedding import Qwen3VLEmbedder

    
class Qwen3VLEmbedding(T2VRetModelBase):
    def __init__(self, config: T2VRetConfig):
        self.config = config
        self.model = self._load_model(self.config.model_path)
        
    def _load_model(self, model_path: str):
        resolved_path = Path(model_path) if os.path.isdir(model_path) else model_path
        
        model = Qwen3VLEmbedder(
            model_name_or_path=resolved_path,
            max_frames=self.config.temporal_size,
            fps=self.config.temporal_size,
            device_map=self.config.device,
        )
        return model

    def _preprocess_video(self, video: Union[torch.Tensor, np.ndarray]) -> List[Image.Image]:
        video = self._to_numpy_frames(video)
        return [self._to_pil(frame) for frame in video]

    def _to_numpy_frames(self, video: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Return video as [T, H, W, C] uint8 numpy array."""
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()
        if video.ndim == 3:                                          # [H, W, C] → [1, H, W, C]
            video = video[np.newaxis]
        if video.shape[1] in (1, 3, 4) and video.shape[-1] not in (1, 3, 4):
            video = video.transpose(0, 2, 3, 1)                     # [T, C, H, W] → [T, H, W, C]
        if video.dtype != np.uint8:
            scale = 255.0 if video.max() <= 1.0 else 1.0
            video = (video * scale).clip(0, 255).astype(np.uint8)
        return video

    def _to_pil(self, frame: np.ndarray) -> Image.Image:
        """Convert a single [H, W, C] uint8 frame to a PIL Image, resizing if configured."""
        pil_frame = Image.fromarray(frame)
        h, w = self.config.img_size
        if h is not None and w is not None:
            pil_frame = pil_frame.resize((w, h), Image.LANCZOS)
        return pil_frame

    def _preprocess_text(self, text: Union[torch.Tensor, np.ndarray, str]) -> str:
        if isinstance(text, (torch.Tensor, np.ndarray)):
            token_ids = text.tolist()
            return self.model.processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        return str(text)

    def preprocess(
        self,
        video: Union[torch.Tensor, np.ndarray, None],
        text: Union[torch.Tensor, np.ndarray, str, List[str], None],
    ):
        # Normalize video to a flat list
        if video is None:
            video_list = []
        elif isinstance(video, (torch.Tensor, np.ndarray)):
            if video.ndim == 5:
                video_list = list(video)
            else:
                video_list = [video]
        else:
            video_list = list(video)

        # Normalize text to a flat list
        if text is None:
            text_list = []
        elif isinstance(text, (str, torch.Tensor, np.ndarray)):
            text_list = [text]
        else:
            text_list = list(text)

        # Build embedder-compatible inputs
        video_inputs = [{"video": self._preprocess_video(v)} for v in video_list]
        text_inputs  = [{"text": self._preprocess_text(t)}   for t in text_list]

        return video_inputs, text_inputs

    @torch.no_grad()
    def forward(
        self,
        video: Union[torch.Tensor, np.ndarray],
        text: Union[torch.Tensor, np.ndarray, str, List[str]],
    ) -> torch.Tensor:
        """Compute cross-modal similarity, or return raw embeddings for a single modality."""
        video_inputs, text_inputs = self.preprocess(video, text)

        # text first so indices align: embeddings[:num_texts] = text, rest = video
        inputs = text_inputs + video_inputs
        embeddings = self.model.process(inputs)

        if text_inputs and video_inputs:
            num_texts = len(text_inputs)
            text_embeddings  = embeddings[:num_texts]
            video_embeddings = embeddings[num_texts:]
            return text_embeddings @ video_embeddings.T

        # Return only the embeddings if either text inputs or video inputs is not available
        return embeddings
    
 
if __name__ == "__main__":
    # MODEL_PATH = "/home/jordan/jordan/Embedding/models/qwen3_vl_embedding/checkpoint/Qwen3-VL-Embedding-2B-FP8_v2"
    MODEL_PATH = "Qwen/Qwen3-VL-Embedding-2B"
    config = T2VRetConfig(
        model_path=MODEL_PATH,
        device="cuda:0",
        temporal_size=8,
        model_name = "Qwen3VLEmbedding",
        img_size=[None, None],
    )
    
    model = Qwen3VLEmbedding(config)
    
    import numpy as np

    np.random.seed(42)
    dummy_B_S_H_W_C_input = np.random.randint(0, 256, size=(1, 8, 1080, 1920, 3), dtype=np.uint8)
    dummy_text = [
        "a person is playing a guitar",
        "a person is playing a piano"
    ]
    
    sim_score = model(video = dummy_B_S_H_W_C_input,
                  text = dummy_text)

    print(sim_score)