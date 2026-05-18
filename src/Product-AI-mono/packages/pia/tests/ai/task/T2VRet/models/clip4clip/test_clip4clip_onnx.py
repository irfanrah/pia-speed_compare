import os
from collections import deque
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from pia.ai.device import load_model_backend
from pia.ai.model import PiaONNXTensorRTModel, PiaTorchModel
from pia.ai.tasks.T2VRet.models.clip4clip.main import Clip4Clip
from pia.ai.tasks.T2VRet.models.clip4clip.model import CLIP4Clip
from pia.tests.test_config import ASSETS_VIDEO_SAVE_DIR
from tabulate import tabulate


def test_clip4clip_onnx_model(
    video_encoder: PiaONNXTensorRTModel,
    text_encoder: PiaONNXTensorRTModel,
    token_embedder: PiaONNXTensorRTModel,
    torch_model: PiaTorchModel,
):
    """Load ONNX Files and Perform Inference

    This function loads ONNX models for video encoding, text encoding, and token embedding,
    sets up a Torch model for the Clip4Clip task, and performs inference on dummy video and text inputs.
    It preprocesses the inputs, encodes the video and text, and calculates the similarity between
    the video and text representations.

    Args:
        device (Literal["cuda", "cpu", "mps"]): The device to run the inference
            on, either 'cuda' for GPU, 'mps' for MAC, or 'cpu'.

    """
    device = load_model_backend("cuda", type="str")

    # Import `Clip4Clip` class to Utilize its Functions
    clip4clip_main = Clip4Clip(config=torch_model.config)

    # Import `CLIP4Clip` class to Utilize its Functions
    state_dict = torch.load(torch_model.config.model_path, map_location=device)
    clip4clip_model = CLIP4Clip(clip_state_dict=state_dict, task_cfg=torch_model.config)

    # Prepare Dummy Input Video and Text
    video_shape = (1, 12, 244, 244, 3)  # Batch, frames, height, width, channel
    dummy_input_vid = np.random.randint(0, 255, video_shape, dtype=np.uint8)
    dummy_input_txt = "self attention is good choice"

    # Preprocess the Dummy Inputs
    #   dummy_input_video_preprocessed.shape: [1, frame, channel, height, width]
    (
        dummy_input_video_preprocessed,
        dummy_video_mask,
        dummy_txt_ids,
        dummy_txt_mask,
    ) = clip4clip_main.preprocess(video=dummy_input_vid, text=dummy_input_txt)
    # Convert dtype to match the precision to the others
    dummy_txt_mask = dummy_txt_mask.to(dtype=torch.float32)

    # Encode Video
    #   We only consider batch size 1
    #   - Inputs
    #       - dummy_input_video_preprocessed.shape:
    #           (Batch size(1), Number of tiles(1), Sequence length(12), Channel(3), Height(224), Width(224))
    #       - visual_output.shape:
    #           (Number of tiles(1), Sequence length(12), Embedding dimension(512))
    #       - visual_output.shape (after unsqueezing):
    #           (Batch size(1), Number of tiles(1), Sequence length(12), Embedding dimension(512))
    #   - Outputs
    #       - vis_vector.shape: (Batch size(1), Number of tiles(1), Embedding dimension(512))
    (
        batch_size,
        num_tiles,
        sequence_length,
        channel,
        height,
        width,
    ) = dummy_input_video_preprocessed.shape
    dummy_input_video_preprocessed = dummy_input_video_preprocessed.reshape(
        batch_size * num_tiles * sequence_length, channel, height, width
    )
    visual_output = video_encoder(d=dummy_input_video_preprocessed)
    visual_output = visual_output.unsqueeze(0)
    vis_vector = clip4clip_model._mean_pooling_for_similarity_visual(
        visual_output=visual_output, video_mask=dummy_video_mask
    )

    # Encode Text
    token_embedding = token_embedder(d=dummy_txt_ids)
    txt_vector = text_encoder(d=[token_embedding, dummy_txt_mask])

    # Calculate Similarity
    similarity_by_onnx_tensor = clip4clip_model._loose_similarity(
        sequence_output=txt_vector, visual_output=vis_vector
    )
    similarity_by_onnx_np = similarity_by_onnx_tensor.cpu().detach().numpy()

    assert np.all((similarity_by_onnx_np >= 0) & (similarity_by_onnx_np <= 1))

    # Compare the Similarity from ONNX Model and Torch Model
    dummy_input_video_for_torch = dummy_input_vid.copy()
    similarity_by_torch_np = torch_model(video=dummy_input_video_for_torch, text=dummy_input_txt)

    assert np.allclose(similarity_by_onnx_np, similarity_by_torch_np, rtol=1e-03, atol=1e-05)


def test_clip4clip_onnx_sim_matrix(
    video_encoder: PiaONNXTensorRTModel,
    text_encoder: PiaONNXTensorRTModel,
    token_embedder: PiaONNXTensorRTModel,
    torch_model: PiaTorchModel,
    setup_sample_videos: List[str],
):
    device = load_model_backend("cuda", type="str")

    def _fill_zero_dequeue(frames_deque: deque, height: int, width: int, channel: int, dtype: type):
        zero_frame = np.zeros((height, width, channel), dtype=dtype)
        for _ in range(frames_deque.maxlen):
            frames_deque.append(zero_frame)
        return frames_deque

    def _encode_video_with_sliding_window(
        video: cv2.VideoCapture,
        frame_skip: int,
        temporal_size: int,
        clip4clip_main: Clip4Clip,
        clip4clip_model: CLIP4Clip,
    ):
        # Validate Video File
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        assert fps > 0, "Invalid Video File"

        # Prepare Visual Vector Placeholder
        vis_vector_list = []
        vid_mask_list = []

        # Prepare Frame Queue
        frames_deque = deque(maxlen=temporal_size)
        frames_deque = _fill_zero_dequeue(
            frames_deque=frames_deque,
            height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            channel=3,
            dtype=np.uint8,
        )

        # Read Frames
        frame_count = 0
        pbar = tqdm.tqdm(total=total_frame, desc=f"Encoding Windows: Window ID: {frame_count}")
        while True:
            success, frame = video.read()

            # Terminate If No Frame Anymore
            if not success:
                break

            # Skip Frame
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            # Collect Frame
            frames_deque.append(frame)
            input_frames_chunk = np.array(frames_deque)

            # Preprocess the Input Frame Chunk and Captions
            #   Inputs
            #       - input_frames_chunk.shape: (Sequence length, Height, Width, Channel)
            #   Outputs
            #       - vid_preprocessed: (Video batch size=1, Number of tiles, Sequence length, Channel, Height, Width)
            #       - vid_mask: (Video batch size=1, Number of tiles, Sequence length)
            vid_preprocessed, vid_mask = clip4clip_main.video_preprocess(video=input_frames_chunk)

            # Encode Video
            pbar.update(frame_skip)
            # print(f"Encoding Window: Window ID: {frame_count}")
            #   Inputs:
            #       - vid_preprocessed.shape: (Video batch size=1, Number of tiles, Sequence length, Channel, Height, Width)
            #       - vid_preprocessed.shape(after reshaping): (Number of all the tiles, Channel, Height, Width)
            #   Ouputs:
            #       - visual_output: (Number of all the tiles, Embedding dimension)
            #       - visual_output(after unsqueezing): (Video batch size, Number of tiles, Sequence length, Embedding dimension)
            (
                batch_size,
                num_tiles,
                sequence_length,
                channel,
                height,
                width,
            ) = vid_preprocessed.shape
            vid_preprocessed = vid_preprocessed.reshape(
                batch_size * num_tiles * sequence_length, channel, height, width
            )
            visual_output = video_encoder(d=vid_preprocessed)
            visual_output = visual_output.view(batch_size, num_tiles, sequence_length, -1)

            assert visual_output.shape == (1, 1, torch_model.config.temporal_size, 512)

            # Mean Pool the Video Vector
            #   Inputs
            #       - visual_output(after unsqueezing): (Video batch size, Number of tiles, Sequence length, Embedding dimension)
            #       - vid_mask: (Video batch size, Number of tiles, Sequence length)
            #   Outputs
            #       - vis_vector: (Video batch size, Number of tiles, Embedding dimension)
            vis_vector = clip4clip_model._mean_pooling_for_similarity_visual(
                visual_output=visual_output, video_mask=vid_mask
            )

            assert vis_vector.shape == (1, 1, 512)

            vis_vector_list.append(vis_vector)
            vid_mask_list.append(vid_mask)

            # End of Loop
            frame_count += 1

        return vis_vector_list, vid_mask_list

    # Import `Clip4Clip` class to Utilize its Functions
    clip4clip_main = Clip4Clip(config=torch_model.config)

    # Import `CLIP4Clip` class to Utilize its Functions
    state_dict = torch.load(torch_model.config.model_path, map_location=device)
    clip4clip_model = CLIP4Clip(clip_state_dict=state_dict, task_cfg=torch_model.config)

    # Prepare Dummy Input Video and Captions
    video_filepath_list = setup_sample_videos
    dummy_input_vid_list = [
        cv2.VideoCapture(video_filepath) for video_filepath in video_filepath_list
    ]
    dummy_input_txt_list = [
        "A person in white forcibly pulls a person in yellow, who is on the ground.",
        "Someone in black roughly grabs a seated person in white.",
        "A man in red, appearing drunk, is being held up by someone in beige pants.",
    ]

    # Preprocess Captions
    #   Inputs
    #       - len(dummy_input_txt_list): Num captions
    #   Outputs
    #       - txt_ids.shape: [Num captions, Max words]
    #       - txt_mask.shape: [Num captions, Max words]
    txt_ids, txt_mask = clip4clip_main.text_preprocess(texts=dummy_input_txt_list)
    # Convert dtype to match the precision to the ONNX file
    txt_mask = txt_mask.to(dtype=torch.float32)

    # Encode Texts
    token_embedding = token_embedder(d=txt_ids)
    txt_vectors = text_encoder(d=[token_embedding, txt_mask])

    # Process All Videos
    num_caps = txt_vectors.shape[0]
    sim_matrix = torch.empty((num_caps, 0), device=device)
    for input_vid in dummy_input_vid_list:
        # Encode a Video Windows
        #   Inputs
        #       - `input_vid`: cv2.VideoCapture
        #   Outputs
        #       - `vis_vector_list`: [vis_vector(Window 0), vis_vector(Window 1), ...]
        #       - `vid_mask_list`: [vid_mask(Window 0), vid_mask(Window 1), ...]
        #       - Window n: frame n - frame (n + temporal size - 1)
        #           - Example: The window assuming the temporal size is 12
        #               - Window 0: frame 0 - frame 11
        #               - Window 1: frame 1 - frame 12
        #               - ...
        vis_vector_list, vid_mask_list = _encode_video_with_sliding_window(
            video=input_vid,
            frame_skip=torch_model.config.frame_skip,
            temporal_size=torch_model.config.temporal_size,
            clip4clip_main=clip4clip_main,
            clip4clip_model=clip4clip_model,
        )

        # Calculate Similarity Scores
        # num_windows = len(vis_vector_list)
        sim_scores_for_video = torch.empty((num_caps, 0), device=device)  # Initialize
        for window_count, (vis_vector, vid_mask) in tqdm.tqdm(
            enumerate(zip(vis_vector_list, vid_mask_list))
        ):
            # print(f"Calculating Simialrity Scores: Window Count: {window_count}/{num_windows}")
            # Inputs
            #   - txt_vectors.shape: [Num captions, 512]
            #   - vis_vector.shape: [1, 512]
            #   - vid_mask: [1, Temporal Size]
            # Outputs
            #   - sim_scores_for_window.shape: [Num captions, 1]
            #   - sim_scores_for_video.shape: [Num captions, Num windows]
            similarity_scores_for_window = clip4clip_model._loose_similarity(
                sequence_output=txt_vectors,
                visual_output=vis_vector,
            )
            sim_scores_for_video = torch.cat(
                (sim_scores_for_video, similarity_scores_for_window), dim=1
            )

        # Aggregate Similarity Scores into Representitive One Value(Score) for Each Video
        #   Inputs
        #       - sim_scores_for_video.shape: [Num captions, Num windows]
        #   Outputs
        #       - sim_score_for_video.shape: [Num captions, 1]
        sim_score_for_video = sim_scores_for_video.max(dim=1).values.unsqueeze(1)

        # Collect Similarity Scores for All Videos
        #   Inputs
        #       - sim_matrix.shape: [Num captions, Num Videos Processed]
        #       - sim_score_for_video.shape: [Num captions, 1]
        sim_matrix = torch.cat((sim_matrix, sim_score_for_video), dim=1)

    # Save Similarity Matrix
    video_filename_list = [
        os.path.basename(video_filepath) for video_filepath in video_filepath_list
    ]
    df_sim_matrix = pd.DataFrame(
        sim_matrix.cpu().numpy(),
        columns=video_filename_list,
        index=dummy_input_txt_list,
    )
    sim_matrix_filepath = os.path.join(ASSETS_VIDEO_SAVE_DIR, "outputs/sim_matrix.csv")
    os.makedirs(os.path.dirname(sim_matrix_filepath), exist_ok=True)
    df_sim_matrix.to_csv(sim_matrix_filepath)
    print(tabulate(df_sim_matrix, headers="keys"))
    print(f"Saved Similarity Matrix: {sim_matrix_filepath}")


def test_get_onnx_vis_vector(
    video_encoder: PiaONNXTensorRTModel,
    setup_sample_videos
):
    video_filepath_list = setup_sample_videos
    vid = cv2.VideoCapture(video_filepath_list[0])
    ret, frame = vid.read()

    assert ret

    time_frame = np.expand_dims(frame, axis=0)
    tile_size = "S"  # Tiled_num = 19

    vis_vector = video_encoder.get_c4c_onnx_vis_vector(
        frame=time_frame,
        tile_size=tile_size,
    )

    assert vis_vector.shape == (1, 19, 512)

    tile_size = None  # Tiled_num = 1

    vis_vector = video_encoder.get_c4c_onnx_vis_vector(
        frame=time_frame,
        tile_size=tile_size,
    )

    assert vis_vector.shape == (1, 1, 512)
