from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class RealFrankaDataset(BaseImageDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 shape_meta=None,
                 dataset_path=None,
                 max_train_episodes=None):
        super().__init__()

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            # keys=['side_image', 'wrist_image', 'state', 'action']
            keys=['side_image', 'state', 'action']
        )

        # 构造训练 mask
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        # 构造采样器
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        # 参数保存
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'state': self.replay_buffer['state']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['side_image'] = get_image_range_normalizer()
        # normalizer['wrist_image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # 提取低维观测
        agent_pos = sample['state'].astype(np.float32)

        # 图像：HWC -> CHW & float32 [0,1]
        side_image = np.moveaxis(sample['side_image'], -1, 1).astype(np.float32) / 255.0
        # wrist_image = np.moveaxis(sample['wrist_image'], -1, 1).astype(np.float32) / 255.0

        data = {
            'obs': {
                'side_image': side_image,
                # 'wrist_image': wrist_image,
                'state': agent_pos,
            },
            'action': sample['action'].astype(np.float32)
        }
        return data

    def __getitem__(self, idx: int):
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


if __name__ == "__main__":
    import os
    zarr_path = os.path.expanduser('data/ts_place_blocks_50_demos.zarr')
    dataset = RealFrankaDataset(zarr_path, horizon=16)
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]

    print("Keys in obs:", sample['obs'].keys())  # 这里应该是 dict_keys(['side_image', 'wrist_image', 'agent_pos'])
    print("Shape of agent_pos:", sample['obs']['state'].shape)
    # sample = dataset[1]
    # print(f"Sample data: {sample}")
