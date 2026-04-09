"""Active Matter dataset loader — reads HDF5 files directly with h5py."""

import os
import re
import glob
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset


class ActiveMatterDataset(Dataset):
    """
    Loads active_matter samples directly from HDF5 files.

    Each sample: (16, 11, 224, 224) float32 tensor + labels (alpha, zeta).
    """

    SPLIT_MAP = {"val": "validation"}

    def __init__(self, data_dir, split="train"):
        """
        Args:
            data_dir: Path to the active_matter dataset root.
            split: One of 'train', 'val', 'test'.
        """
        data_dir = os.path.expandvars(data_dir)
        hf_split = self.SPLIT_MAP.get(split, split)

        # Find the HDF5 files directory
        split_dir = None
        for candidate in [
            os.path.join(data_dir, "data", hf_split),
            os.path.join(data_dir, "data", split),
            os.path.join(data_dir, hf_split),
            os.path.join(data_dir, split),
        ]:
            if os.path.isdir(candidate):
                split_dir = candidate
                break

        if split_dir is None:
            raise FileNotFoundError(
                f"Could not find split '{split}' in {data_dir}. "
                f"Expected subdirectory like data/{hf_split}/ or {hf_split}/"
            )

        self.files = sorted(glob.glob(os.path.join(split_dir, "*.hdf5")))
        if not self.files:
            raise FileNotFoundError(f"No .hdf5 files found in {split_dir}")

        print(f"[{split}] Found {len(self.files)} HDF5 files in {split_dir}")

        # Discover field names from first file
        with h5py.File(self.files[0], "r") as f:
            self._field_keys = sorted(f.keys())
            self._attrs = dict(f.attrs)
            # Check if file has top-level datasets or groups
            first_key = self._field_keys[0]
            sample_obj = f[first_key]
            if isinstance(sample_obj, h5py.Group):
                # Grouped structure: each top-level key is a sample
                self._mode = "grouped"
            else:
                # Flat structure: each file is one sample, keys are fields
                self._mode = "flat"

        print(f"  Mode: {self._mode}, Fields: {self._field_keys}")

    def __len__(self):
        return len(self.files)

    def _parse_labels_from_filename(self, filepath):
        """Extract alpha and zeta from filename like active_matter_L_10.0_zeta_1.0_alpha_-1.0.hdf5"""
        basename = os.path.basename(filepath)
        alpha_match = re.search(r'alpha_([-\d.]+)', basename)
        zeta_match = re.search(r'zeta_([-\d.]+)', basename)
        alpha = float(alpha_match.group(1)) if alpha_match else 0.0
        zeta = float(zeta_match.group(1)) if zeta_match else 0.0
        return alpha, zeta

    def _load_flat(self, f):
        """Load from flat HDF5: keys are field names, each is a dataset."""
        fields = []
        for key in sorted(f.keys()):
            data = f[key][:]  # Read full array
            data = data.astype(np.float32)

            if data.ndim == 3:       # (T, H, W) → (T, 1, H, W)
                data = data[:, np.newaxis, :, :]
            elif data.ndim == 4:     # (T, C, H, W) — keep as is
                pass
            elif data.ndim == 5:     # (T, d1, d2, H, W) → flatten tensor dims
                T, d1, d2, H, W = data.shape
                data = data.reshape(T, d1 * d2, H, W)

            fields.append(data)

        # Stack along channel dim: (T, 11, H, W)
        x = np.concatenate(fields, axis=1)
        return x

    def __getitem__(self, idx):
        filepath = self.files[idx]

        with h5py.File(filepath, "r") as f:
            x = self._load_flat(f)

            # Try to get labels from HDF5 attributes first
            alpha = float(f.attrs.get("alpha", 0.0))
            zeta = float(f.attrs.get("zeta", 0.0))

            # If attrs are zero/missing, parse from filename
            if alpha == 0.0 and zeta == 0.0:
                alpha, zeta = self._parse_labels_from_filename(filepath)

        x = torch.from_numpy(x)  # (T, C, H, W)

        labels = {
            "alpha": torch.tensor(alpha, dtype=torch.float32),
            "zeta": torch.tensor(zeta, dtype=torch.float32),
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
