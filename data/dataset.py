"""Active Matter dataset loader — reads HDF5 files directly with h5py.

HDF5 structure per file (one per parameter combo):
  scalars/alpha, scalars/zeta          — physical parameters
  t0_fields/concentration              — (N_traj, 81, 256, 256)         → 1 channel
  t1_fields/velocity                   — (N_traj, 81, 256, 256, 2)     → 2 channels
  t2_fields/D (orientation tensor)     — (N_traj, 81, 256, 256, 2, 2)  → 4 channels
  t2_fields/E (strain-rate tensor)     — (N_traj, 81, 256, 256, 2, 2)  → 4 channels
                                                                Total: 11 channels

Each trajectory has 81 time steps at 256×256.
We extract sliding windows of 16 frames and center-crop to 224×224.
  → 45 files × ~3 traj × ~66 windows ≈ 8,750 train samples
"""

import os
import glob
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset


class ActiveMatterDataset(Dataset):
    """
    Loads active_matter samples from HDF5 files.

    Each sample: (16, 11, 224, 224) float32 tensor + labels (alpha, zeta).

    Indexing: Each item corresponds to (file_idx, trajectory_idx, temporal_window_start).
    """

    SPLIT_MAP = {"val": "validation"}

    def __init__(self, data_dir, split="train", n_frames=16, spatial_crop=224, temporal_stride=1):
        """
        Args:
            data_dir: Path to the active_matter dataset root.
            split: One of 'train', 'val', 'test'.
            n_frames: Number of time steps per sample window.
            spatial_crop: Crop spatial dims to this size (center crop).
            temporal_stride: Stride between consecutive windows.
        """
        data_dir = os.path.expandvars(data_dir)
        hf_split = self.SPLIT_MAP.get(split, split)

        # Find the HDF5 files directory — try multiple naming conventions
        split_dir = None
        candidates = [hf_split, split, "valid"] if split == "val" else [hf_split, split]
        for prefix in ["data", ""]:
            for name in candidates:
                path = os.path.join(data_dir, prefix, name) if prefix else os.path.join(data_dir, name)
                if os.path.isdir(path):
                    split_dir = path
                    break
            if split_dir:
                break

        if split_dir is None:
            raise FileNotFoundError(
                f"Could not find split '{split}' in {data_dir}. "
                f"Tried names: {candidates}"
            )

        self.files = sorted(glob.glob(os.path.join(split_dir, "*.hdf5")))
        if not self.files:
            raise FileNotFoundError(f"No .hdf5 files found in {split_dir}")

        self.n_frames = n_frames
        self.spatial_crop = spatial_crop

        # Build index: list of (file_idx, traj_idx, t_start)
        self.samples = []
        for file_idx, filepath in enumerate(self.files):
            with h5py.File(filepath, "r") as f:
                n_traj = int(f.attrs.get("n_trajectories", 1))
                n_time = f["t0_fields/concentration"].shape[1]  # 81

            n_windows = (n_time - n_frames) // temporal_stride + 1
            for traj_idx in range(n_traj):
                for t_start in range(0, n_time - n_frames + 1, temporal_stride):
                    self.samples.append((file_idx, traj_idx, t_start))

        print(f"[{split}] {len(self.files)} HDF5 files → "
              f"{len(self.samples)} samples "
              f"({n_frames} frames, {spatial_crop}×{spatial_crop} crop, stride={temporal_stride})")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read_label(f, name):
        """Read a scalar label from HDF5, trying multiple locations."""
        import re
        # Try scalars/ group first
        key = f"scalars/{name}"
        if key in f:
            val = f[key][()]
            return float(val.flat[0]) if hasattr(val, 'flat') else float(val)
        # Try file attributes
        if name in f.attrs:
            return float(f.attrs[name])
        # Parse from filename
        m = re.search(rf'{name}_([-\d.]+)', os.path.basename(f.filename))
        return float(m.group(1)) if m else 0.0

    def __getitem__(self, idx):
        file_idx, traj_idx, t_start = self.samples[idx]
        t_end = t_start + self.n_frames

        with h5py.File(self.files[file_idx], "r") as f:
            # Read labels — try scalars/ datasets first, then attributes, then filename
            alpha = self._read_label(f, "alpha")
            zeta = self._read_label(f, "zeta")

            # Read fields for this trajectory and time window
            # concentration: (N, T, H, W) → (n_frames, 1, H, W)
            conc = f["t0_fields/concentration"][traj_idx, t_start:t_end]  # (16, 256, 256)

            # velocity: (N, T, H, W, 2) → (n_frames, 2, H, W)
            vel = f["t1_fields/velocity"][traj_idx, t_start:t_end]  # (16, 256, 256, 2)

            # D (orientation): (N, T, H, W, 2, 2) → (n_frames, 4, H, W)
            D = f["t2_fields/D"][traj_idx, t_start:t_end]  # (16, 256, 256, 2, 2)

            # E (strain-rate): (N, T, H, W, 2, 2) → (n_frames, 4, H, W)
            E = f["t2_fields/E"][traj_idx, t_start:t_end]  # (16, 256, 256, 2, 2)

        # Reshape to (T, C, H, W)
        T, H, W = conc.shape
        conc = conc[:, np.newaxis, :, :]                    # (T, 1, H, W)
        vel = vel.transpose(0, 3, 1, 2)                     # (T, 2, H, W)
        D = D.reshape(T, H, W, 4).transpose(0, 3, 1, 2)    # (T, 4, H, W)
        E = E.reshape(T, H, W, 4).transpose(0, 3, 1, 2)    # (T, 4, H, W)

        x = np.concatenate([conc, vel, D, E], axis=1)       # (T, 11, H, W)

        # Center crop: 256 → 224
        if self.spatial_crop < H:
            offset = (H - self.spatial_crop) // 2
            x = x[:, :, offset:offset + self.spatial_crop, offset:offset + self.spatial_crop]

        x = torch.from_numpy(x.astype(np.float32))

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
