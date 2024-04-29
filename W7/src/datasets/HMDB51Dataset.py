""" Dataset class for HMDB51 dataset. """
import os
import random
from enum import Enum
import numpy as np

from glob import glob, escape
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

class HMDB51Dataset(Dataset):
    """
    Dataset class for HMDB51 dataset.
    """

    class Split(Enum):
        """
        Enum class for dataset splits.
        """
        TEST_ON_SPLIT_1 = 1
        TEST_ON_SPLIT_2 = 2
        TEST_ON_SPLIT_3 = 3

    class Regime(Enum):
        """
        Enum class for dataset regimes.
        """
        TRAINING = 1
        TESTING = 2
        VALIDATION = 3

    CLASS_NAMES = [
        "brush_hair", "catch", "clap", "climb_stairs", "draw_sword", "drink", 
        "fall_floor", "flic_flac", "handstand", "hug", "kick", "kiss", "pick", 
        "pullup", "push", "ride_bike", "run", "shoot_ball", "shoot_gun", "situp", 
        "smoke", "stand", "sword", "talk", "turn", "wave", 
        "cartwheel", "chew", "climb", "dive", "dribble", "eat", "fencing", 
        "golf", "hit", "jump", "kick_ball", "laugh", "pour", "punch", "pushup", 
        "ride_horse", "shake_hands", "shoot_bow", "sit", "smile", "somersault", 
        "swing_baseball", "sword_exercise", "throw", "walk"
    ]


    def __init__(
        self, 
        videos_dir: str, 
        annotations_dir: str, 
        split: Split, 
        regime: Regime, 
        clip_length: int, 
        crop_size: int, 
        temporal_stride: int,
        clips_per_video: int,
        crops_per_clip: int,
        tsn_k: int,
        deterministic: bool,
        model_name: str,
        mode: str
    ) -> None:
        """
        Initialize HMDB51 dataset.

        Args:
            videos_dir (str): Directory containing video files.
            annotations_dir (str): Directory containing annotation files.
            split (Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
            regime (Regimes): Dataset regime (TRAINING, TESTING, VALIDATION).
            split (Splits): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
            clip_length (int): Number of frames of the clips.
            crop_size (int): Size of spatial crops (squares).
            temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.
            clips_per_video (int): Number of clips to sample from each video.
            crops_per_clip (int): Number of crops to sample from each clip.
            tsn_k (int): Number of segments for Temporal Segment Network (TSN).
            deterministic (bool): Whether to use deterministic sampling (if False, TSN is used)
        """
        self.videos_dir = videos_dir
        self.annotations_dir = annotations_dir
        self.split = split
        self.regime = regime
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.temporal_stride = temporal_stride
        self.clips_per_video = clips_per_video
        self.crops_per_clip = crops_per_clip
        self.tsn_k = tsn_k
        self.deterministic = deterministic
        self.model_name = model_name
        self.mode = mode

        self.annotation = self._read_annotation()
        self.transform = self._create_transform()
        
        # Force clips_per_video to be 1 if TSN is used
        if not self.deterministic and self.tsn_k > 1:
            self.clips_per_video = 1

    def _standardized_crop(self, transform):
        return v2.Compose([
            v2.Resize(self.crop_size), # Shortest side of the frame to be resized to the given size
            transform,
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _positional_crop(self, position):
        CENTER_LEFT_TRANSFORM = self._standardized_crop(v2.Lambda(lambda img: v2.functional.crop(img, img.shape[1] // 2 - self.crop_size // 2, 0, self.crop_size, self.crop_size)))
        CENTER_RIGHT_TRANSFORM = self._standardized_crop(v2.Lambda(lambda img: v2.functional.crop(img, img.shape[1] // 2 - self.crop_size // 2, img.shape[2] - self.crop_size, self.crop_size, self.crop_size)))
        CENTER_CENTER_TRANSFORM = self._standardized_crop(v2.CenterCrop(self.crop_size))
        TOP_LEFT_TRANSFORM = self._standardized_crop(v2.Lambda(lambda img: v2.functional.crop(img, 0, 0, self.crop_size, self.crop_size)))
        TOP_RIGHT_TRANSFORM = self._standardized_crop(v2.Lambda(lambda img: v2.functional.crop(img, 0, img.shape[2] - self.crop_size, self.crop_size, self.crop_size)))
        BOTTOM_LEFT_TRANSFORM = self._standardized_crop(v2.Lambda(lambda img: v2.functional.crop(img, img.shape[1] - self.crop_size, 0, self.crop_size, self.crop_size)))
        BOTTOM_RIGHT_TRANSFORM = self._standardized_crop(v2.Lambda(lambda img: v2.functional.crop(img, img.shape[1] - self.crop_size, img.shape[2] - self.crop_size, self.crop_size, self.crop_size)))

        if position == 'center_left':
            return CENTER_LEFT_TRANSFORM
        elif position == 'center_right':
            return CENTER_RIGHT_TRANSFORM
        elif position == 'center_center':
            return CENTER_CENTER_TRANSFORM
        elif position == 'top_left':
            return TOP_LEFT_TRANSFORM
        elif position == 'top_right':
            return TOP_RIGHT_TRANSFORM
        elif position == 'bottom_left':
            return BOTTOM_LEFT_TRANSFORM
        elif position == 'bottom_right':
            return BOTTOM_RIGHT_TRANSFORM
        else:
            raise ValueError(f"Invalid crop position: {position}")

    def _read_annotation(self) -> pd.DataFrame:
        """
        Read annotation files.

        Returns:
            pd.DataFrame: Dataframe containing video annotations.
        """
        split_suffix = "_test_split" + str(self.split.value) + ".txt"

        annotation = []
        for class_name in HMDB51Dataset.CLASS_NAMES:
            annotation_file = os.path.join(self.annotations_dir, class_name + split_suffix)
            df = pd.read_csv(annotation_file, sep=" ").dropna(axis=1, how='all') # drop empty columns
            df.columns = ['video_name', 'train_or_test']
            df = df[df.train_or_test == self.regime.value]
            df = df.rename(columns={'video_name': 'video_path'})
            df['video_path'] = os.path.join(self.videos_dir, class_name, '') + df['video_path'].replace('\.avi$', '', regex=True)
            df = df.rename(columns={'train_or_test': 'class_id'})
            df['class_id'] = HMDB51Dataset.CLASS_NAMES.index(class_name)
            annotation += [df]

        return pd.concat(annotation, ignore_index=True)


    def _create_transform(self) -> [v2.Compose]:
        """
        Create transform based on the dataset regime.

        Returns:
            v2.Compose: Transform for the dataset.
        """
        if self.regime == HMDB51Dataset.Regime.TRAINING:
            if self.model_name == "movinet_a0":
                return [v2.Compose([
                    # Transform to float32 and normalize between 0 and 1

                    v2.Resize((200, 200)),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                    v2.RandomResizedCrop(self.crop_size),
                ])]
            else:
                if self.mode == "rgb":
                    return [v2.Compose([
                        v2.RandomResizedCrop(self.crop_size),
                        v2.RandomHorizontalFlip(p=0.5),
                        # v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])]
                elif self.mode == "flow":
                    return [v2.Compose([
                        v2.RandomResizedCrop(self.crop_size),
                        # v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        v2.ToDtype(torch.float32, scale=True),
                    ])]
        else:
            t = []
            if self.crops_per_clip == 0 or self.crops_per_clip == 1 or self.crops_per_clip == 3 or self.crops_per_clip == 5:
                t.append(self._positional_crop('center_center'))
            if self.crops_per_clip == 2 or self.crops_per_clip == 3:
                t.append(self._positional_crop('center_left'))
                t.append(self._positional_crop('center_right'))
            if self.crops_per_clip == 4 or self.crops_per_clip == 5:
                t.append(self._positional_crop('top_left'))
                t.append(self._positional_crop('top_right'))
                t.append(self._positional_crop('bottom_left'))
                t.append(self._positional_crop('bottom_right'))
            return t


    def get_num_classes(self) -> int:
        """
        Get the number of classes.

        Returns:
            int: Number of classes.
        """
        return len(HMDB51Dataset.CLASS_NAMES)


    def __len__(self) -> int:
        """
        Get the length (number of videos) of the dataset.

        Returns:
            int: Length (number of videos) of the dataset.
        """
        return len(self.annotation)
    
    # ? Original function
    # def __getitem__(self, idx: int) -> tuple:
    #     """
    #     Get item (video) from the dataset.

    #     Args:
    #         idx (int): Index of the item (video).

    #     Returns:
    #         tuple: Tuple containing video, label, and video path.
    #     """
    #     df_idx = self.annotation.iloc[idx]

    #     # Get video path from the annotation dataframe and check if it exists
    #     video_path = df_idx['video_path']
    #     assert os.path.exists(video_path)

    #     # Read frames' paths from the video
    #     frame_paths = sorted(glob(os.path.join(escape(video_path), "*.jpg"))) # get sorted frame paths
    #     video_len = len(frame_paths)

    #     if video_len <= self.clip_length * self.temporal_stride:
    #         # Not enough frames to create the clip
    #         clip_begin, clip_end = 0, video_len
    #     else:
    #         # Randomly select a clip from the video with the desired length (start and end frames are inclusive)
    #         clip_begin = random.randint(0, max(video_len - self.clip_length * self.temporal_stride, 0))
    #         clip_end = clip_begin + self.clip_length * self.temporal_stride

    #     # Read frames from the video with the desired temporal subsampling
    #     video = None
    #     for i, path in enumerate(frame_paths[clip_begin:clip_end:self.temporal_stride]):
    #         frame = read_image(path)  # (C, H, W)
    #         if video is None:
    #             video = torch.zeros((self.clip_length, 3, frame.shape[1], frame.shape[2]), dtype=torch.uint8)
    #         video[i] = frame

    #     # Get label from the annotation dataframe and make sure video was read
    #     label = df_idx['class_id']
    #     assert video is not None

    #     return video, label, video_path

    def _get_segments(self, frame_paths, video_len):
        clips = []
        if video_len >= self.clip_length * self.temporal_stride:
            # Calculate the indices for starting frames for evenly spaced clips
            max_start_index =  video_len - self.clip_length * self.temporal_stride
            if self.deterministic:
                clip_starts = np.linspace(0, max_start_index, self.clips_per_video, dtype=int, endpoint=False)
            else:
                if max_start_index > 0:
                    clip_starts = np.random.randint(0, max_start_index, self.clips_per_video)
                else:
                    clip_starts = [0] * self.clips_per_video

            # Collect the clips
            for start in clip_starts:
                clip_frames = [read_image(frame_paths[start + i * self.temporal_stride]) for i in range(self.clip_length)]
                clip = torch.stack(clip_frames)
                clips.append(clip)
        else:
            # If not enough frames for desired number of clips, repeat the available frames
            repetition_factor = (self.clips_per_video * self.clip_length * self.temporal_stride + video_len - 1) // video_len
            repeated_frame_paths = frame_paths * repetition_factor

            clips = []
            for i in range(self.clips_per_video):
                real_i = i
                if not self.deterministic:
                    real_i = random.randint(0, video_len)
                clip_frames = []
                for j in range(self.clip_length):
                    frame_idx = (real_i * self.clip_length + j) * self.temporal_stride % len(repeated_frame_paths)
                    frame = read_image(repeated_frame_paths[frame_idx])  # Read the frame image
                    clip_frames.append(frame)
                clips.append(torch.stack(clip_frames))  # Stack the frames to form a clip tensor
        return clips

    # ? Modified function
    def __getitem__(self, idx: int) -> tuple:
        """
        Get item (video) from the dataset.

        Args:
            idx (int): Index of the item (video).

        Returns:
            tuple: Tuple containing videos, label, and video path.
        """
        df_idx = self.annotation.iloc[idx]

        # Get video path from the annotation dataframe and check if it exists
        video_path = df_idx['video_path']
        assert os.path.exists(video_path), f"Video path does not exist: {video_path}"

        # Read frames' paths from the video
        if self.mode == "rgb":
            frame_paths = sorted(glob(os.path.join(escape(video_path), "*.jpg")))  # get sorted frame paths
        elif self.mode == "flow":
            frame_paths = sorted(glob(os.path.join(escape(video_path), "*.png")))  # get sorted flow paths
        video_len = len(frame_paths)

        if self.tsn_k < 2:
            clips = self._get_segments(frame_paths, video_len)
        else:
            real_video_len = video_len // self.tsn_k
            clips = []
            for offset in range(self.tsn_k):
                f_paths = frame_paths[offset * real_video_len: (offset + 1) * real_video_len]
                clips.extend(self._get_segments(f_paths, real_video_len))


        # Stack all clips along the first dimension to get a tensor of shape (num_clips, clip_length, C, H, W)
        clips_tensor = torch.stack(clips, dim=0)

        # Get label from the annotation dataframe
        label = df_idx['class_id']

        return clips_tensor, label, video_path

    # ? Original function
    # def collate_fn(self, batch: list) -> dict:
    #     """
    #     Collate function for creating batches.

    #     Args:
    #         batch (list): List of samples.

    #     Returns:
    #         dict: Dictionary containing batched clips, labels, and paths.
    #     """
    #     # [(clip1, label1, path1), (clip2, label2, path2), ...]
    #     #   -> ([clip1, clip2, ...], [label1, label2, ...], [path1, path2, ...])
    #     unbatched_clips, unbatched_labels, paths = zip(*batch)
 
    #     # Apply transformation and permute dimensions: (T, C, H, W) -> (C, T, H, W)
    #     transformed_clips = [self.transform(clip).permute(1, 0, 2, 3) for clip in unbatched_clips]
    #     # Concatenate clips along the batch dimension:
    #     # B * [(C, T, H, W)] -> B * [(1, C, T, H, W)] -> (B, C, T, H, W)
    #     batched_clips = torch.cat([d.unsqueeze(0) for d in transformed_clips], dim=0)

    #     return dict(
    #         clips=batched_clips, # (B, C, T, H, W)
    #         labels=torch.tensor(unbatched_labels), # (K,)
    #         paths=paths  # no need to make it a tensor
    #     )

    # ? Modified function
    def collate_fn(self, batch: list) -> dict:
        """
        Collate function for creating batches.

        Args:
            batch (list): List of samples.

        Returns:
            dict: Dictionary containing batched clips, labels, and paths.
        """
        # Unpack the batch. Each element in `batch` is (clips_tensor, label, video_path)
        # with `clips_tensor` being of shape (num_clips_per_video, clip_length, C, H, W).
        batched_clips, batched_labels, paths = [], [], []

        for clips_tensor, label, video_path in batch:
            # clips_tensor is (num_clips_per_video, clip_length, C, H, W)
            # We will transform each clip individually and then stack them.
            clips = []
            for clip in clips_tensor:
                for transform in self.transform:
                    transformed_clip = transform(clip)
                    if self.model_name != "resnet50":
                        transformed_clip = transformed_clip.permute(1, 0, 2, 3)
                        clips.append(transformed_clip.unsqueeze(0))
                    else:
                        clips.append(transformed_clip)
            clips = torch.cat(clips, dim=0)
            batched_clips.extend([clips.unsqueeze(0)])  # Add to the list of clips
            if self.model_name != "resnet50":
                batched_labels.extend([label])  # Repeat label for each clip
                paths.extend([video_path])  # Repeat path for each clip
            else:
                batched_labels.extend([label] * clips.size(0))
                paths.extend([video_path] * clips.size(0))

        # Concatenate all the clips along the batch dimension
        # (num_videos * num_clips_per_video, C, T, H, W)
        batched_clips = torch.cat(batched_clips, dim=0)

        # Convert labels to tensor
        # (num_videos * num_clips_per_video,)
        batched_labels = torch.tensor(batched_labels, dtype=torch.long)

        return dict(
            clips=batched_clips,
            labels=batched_labels,
            paths=paths
        )
