from typing import Union

import torch
from pia.utils.exception.model_handler import (
    validate_text_embedding_vector,
    validate_video_embedding_vector,
)
from torch import nn

from .module_clip import CLIP, convert_weights
from .module_cross import Transformer as TransformerClip


def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


def update_attr(
    target_name,
    target_config,
    target_attr_name,
    source_config,
    source_attr_name,
    default_value=None,
):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(
                target_config,
                target_attr_name,
                getattr(source_config, source_attr_name),
            )
    return target_config


class CLIP4Clip(nn.Module):
    def __init__(self, clip_state_dict, task_cfg):
        super().__init__()
        self.task_config = task_cfg
        self.ignore_video_index = -1

        assert (
            self.task_config.max_words + self.task_config.temporal_size
            <= task_cfg.max_position_embeddings
        )

        self._stage_one = True
        self._stage_two = False

        self.loose_type = False
        if self._stage_one and check_attr("loose_type", self.task_config):
            self.loose_type = True

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "clip.visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["clip.visual.conv1.weight"].shape[0]
            vision_layers = len(
                [
                    k
                    for k in clip_state_dict.keys()
                    if k.startswith("clip.visual.") and k.endswith(".attn.in_proj_weight")
                ]
            )
            vision_patch_size = clip_state_dict["clip.visual.conv1.weight"].shape[-1]
            grid_size = round(
                (clip_state_dict["clip.visual.positional_embedding"].shape[0] - 1) ** 0.5
            )
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [
                len(
                    {
                        k.split(".")[2]
                        for k in clip_state_dict
                        if k.startswith(f"clip.visual.layer{b}")
                    }
                )
                for b in [1, 2, 3, 4]
            ]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["clip.visual.layer1.0.conv1.weight"].shape[0]
            output_width = round(
                (clip_state_dict["clip.visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
            )
            vision_patch_size = None
            assert (
                output_width**2 + 1
                == clip_state_dict["clip.visual.attnpool.positional_embedding"].shape[0]
            )
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["clip.text_projection"].shape[1]
        context_length = clip_state_dict["clip.positional_embedding"].shape[0]
        vocab_size = clip_state_dict["clip.token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["clip.ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(
            {k.split(".")[3] for k in clip_state_dict if k.startswith("clip.transformer.resblocks")}
        )

        self.linear_patch = "2d"
        if hasattr(task_cfg, "linear_patch"):
            self.linear_patch = task_cfg.linear_patch

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        self.clip = CLIP(
            embed_dim,
            image_resolution,
            vision_layers - cut_top_layer,
            vision_width,
            vision_patch_size,
            context_length,
            vocab_size,
            transformer_width,
            transformer_heads,
            transformer_layers - cut_top_layer,
            linear_patch=self.linear_patch,
        ).float()

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = "meanP"
        if hasattr(task_cfg, "sim_header"):
            self.sim_header = task_cfg.sim_header
        if self.sim_header == "tightTransf":
            assert self.loose_type is False

        task_cfg.max_position_embeddings = context_length

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(
                task_cfg.max_position_embeddings, task_cfg.hidden_size
            )
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(
                width=transformer_width,
                layers=self.task_config.cross_num_hidden_layers,
                heads=transformer_heads,
            )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(
                input_size=task_cfg.hidden_size,
                hidden_size=task_cfg.hidden_size,
                batch_first=True,
                bidirectional=False,
                num_layers=1,
            )

        self.apply(self.init_weights)

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.task_config.initializer_range)
        elif isinstance(module, LayerNorm):
            if "beta" in dir(module) and "gamma" in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        sequence_output, visual_output = self.get_sequence_visual_output(
            input_ids,
            token_type_ids,
            attention_mask,
            video,
            video_mask,
            shaped=True,
            video_frame=video_frame,
        )
        return sequence_output, visual_output

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        # FIXME: `token_type_ids` and `attention_mask` are not used
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        sequence_hidden = self.clip.encode_text(input_ids).float()

        return sequence_hidden

    def get_visual_output(
        self,
        video: torch.Tensor,
        video_mask: Union[None, torch.Tensor],
    ):
        """
        Encode video to embedding vector.

        This method performs the following steps:

        1. Encodes the video frames using `self.clip.encode_image()` which
           processes each frame independently. The input video tensor is
           reshaped to have the shape (Number of all the  tiles, Channel,
           Height, Width). After encoding, the output embedding vectors
           `visual_output` have the shape (Number of all the tiles, Embedding
           dimension).

        2. Reshapes the `visual_output` tensor back to include batch size,
           number of tiles, and sequence length. The reshaped tensor has the
           shape (Batch size, Number of tiles, Sequence length, Embedding
           dimension).

        3. Normalizes the `visual_output` tensor along the embedding dimension.

        4. Applies mean pooling for sequence-wise using the method
           `self._mean_pooling_for_similarity_visual()`, which converts the
           `visual_output` shape to (Batch size, Number of tiles, Embedding
           dimension).

        Args:
            video (torch.Tensor): Video tensor. Shape: (Batch size, Number of
                tiles, Sequence length, Channel, Height, Width).
            video_mask (Union[None, torch.Tensor]): Video mask tensor. Shape:
                (Batch size, Number of tiles, Sequence length). Defaults to None if
                not provided.

        Returns:
            torch.Tensor: Video embedding vector. Shape: (Batch size, Number of
                tiles, Embedding dimension).
        """
        batch_size, num_tiles, sequence_length, channel, height, width = video.shape

        if video_mask is None:
            # TODO: Check if video_mask is correct when None
            video_mask = torch.zeros(batch_size * num_tiles * sequence_length)

        visual_output = self.clip.encode_image(
            video.reshape(batch_size * num_tiles * sequence_length, channel, height, width),
            video_frame=-1,  # TODO: What is it?
        ).float()

        embedding_dimension = visual_output.size(-1)
        visual_output = visual_output.view(
            batch_size, num_tiles, sequence_length, embedding_dimension
        )

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return visual_output

    def get_sequence_visual_output(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        video,
        video_mask,
    ):
        sequence_output = self.get_sequence_output(
            input_ids, token_type_ids, attention_mask, shaped=True
        )

        visual_output = self.get_visual_output(video, video_mask)

        return sequence_output, visual_output

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.0
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(
            attention_mask_un, dim=1, dtype=torch.float
        )
        return text_out

    def _mean_pooling_for_similarity_visual(
        self,
        visual_output,
        video_mask,
    ):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=2, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.0] = 1.0
        video_out = torch.sum(visual_output, dim=2) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(
        self,
        sequence_output,
        visual_output,
        attention_mask,
        video_mask,
    ):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(
        self, sequence_output: torch.Tensor, visual_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate similarity between text and video embeddings.

        This method calculates the similarity between text and video embeddings
        using the following steps:
            1. Normalize the text and video embeddings.
            2. Calculate the dot product of the normalized text and video
               embeddings.
            3. Apply max similarity strategy along the tiles.

                >> for tile_id in range(num_tiles):
                >>     visual_output_tile = visual_output[:, tile_id, :]
                >>     retrieve_logits_tile = torch.matmul(sequence_output, visual_output_tile.t())
                >>     retrieve_logits.append(retrieve_logits_tile)

                The output of the above code snippet is a list of similarity logits for
                each tile. [retrieve_logits_tile_1, retrieve_logits_tile_2, ...,]
                retrieve_logits_tile_i: Shape: (Number of captions, Number of video batches)

                >> retrieve_logits = torch.stack(retrieve_logits, dim=1)

                The output of the above code snippet is a tensor of similarity logits
                for all the tiles. Shape: (Number of captions, Number of tiles, Number of video batches)

                >> retrieve_logits = torch.max(retrieve_logits, dim=1).values

                The output of the above code snippet is a tensor of similarity logits
                for all the tiles. Shape: (Number of captions, Number of video batches)

        Args:
            sequence_output (torch.Tensor): Text embedding vector. Shape: (Batch size, Embedding dimension).
            visual_output (torch.Tensor): Video embedding vector. Shape: (Batch size, Number of tiles, Embedding dimension).

        Returns:
            torch.Tensor: Similarity logits tensor. Shape: (Number of captions, Number of video batches).

        """
        validate_text_embedding_vector(sequence_output)
        validate_video_embedding_vector(visual_output)

        sequence_output, visual_output = (
            sequence_output.contiguous(),
            visual_output.contiguous(),
        )

        # Normalize each feature vector
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        # Calculate dot products of unit vectors (=cosine similarity)
        batch_size, num_tiles, embedding_dimension = visual_output.shape
        retrieve_logits = []
        for tile_id in range(num_tiles):
            visual_output_tile = visual_output[:, tile_id, :]
            retrieve_logits_tile = torch.matmul(sequence_output, visual_output_tile.t())
            retrieve_logits.append(retrieve_logits_tile)

        # Max similarity strategy along the tiles
        stack_retrieve_logits = torch.stack(retrieve_logits, dim=1)
        max_retrieve_logits = torch.max(stack_retrieve_logits, dim=1).values

        return max_retrieve_logits

    def get_similarity_logits(
        self,
        sequence_output,
        visual_output,
        attention_mask,
        video_mask,
        shaped=False,
        loose_type=False,
    ):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            retrieve_logits = self._loose_similarity(
                sequence_output,
                visual_output,
                attention_mask,
                video_mask,
                sim_header=self.sim_header,
            )
        else:
            raise

        return retrieve_logits, contrastive_direction


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
