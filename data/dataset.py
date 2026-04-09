"""Active Matter dataset for HuggingFace Arrow files."""

import os
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk


class ActiveMatterDataset(Dataset):
    """
    Loads active_matter samples from HuggingFace disk format.

    Each sample: (16, 11, 224, 224) float32 tensor + labels (alpha, zeta).
    """

    FIELD_NAMES = [
        "concentration",              # 1 scalar
        "velocity_x", "velocity_y",   # 2 vector
        "orientation_xx", "orientation_xy", "orientation_yx", "orientation_yy",  # 4 tensor
        "strain_rate_xx", "strain_rate_xy", "strain_rate_yx", "strain_rate_yy",  # 4 tensor
    ]

    def __init__(self, data_dir, split="train", normalize=True):
        """
        Args:
            data_dir: Path to the active_matter dataset root.
            split: One of 'train', 'val', 'test'.
            normalize: Whether to z-score normalize each channel.
        """
        split_dir = os.path.join(data_dir, split)
        self.ds = load_from_disk(split_dir)
        self.normalize = normalize
        self._channel_stats = None  # Lazy-computed

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]

        # Stack all fields → (T, C, H, W)
        fields = []
        for name in self.FIELD_NAMES:
            field = torch.tensor(sample[name], dtype=torch.float32)
            if field.ndim == 3:  # (T, H, W) → (T, 1, H, W)
                field = field.unsqueeze(1)
            fields.append(field)

        x = torch.cat(fields, dim=1)  # (16, 11, 224, 224)

        labels = {
            "alpha": torch.tensor(sample["alpha"], dtype=torch.float32),
            "zeta": torch.tensor(sample["zeta"], dtype=torch.float32),
        }

        return x, labels


def get_dataloaders(data_dir, batch_size=4, num_workers=4):
    """Create train/val/test dataloaders."""
    from torch.utils.data import DataLoader

    loaders = {}
    for split in ["train", "val", "test"]:
        ds = ActiveMatterDataset(data_dir, split=split)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
