#!/usr/bin/env python
"""Inspect the HDF5 structure of an active_matter sample."""

import os
import sys
import glob
import h5py


def inspect_hdf5(path):
    """Print the full structure of an HDF5 file."""
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"Size: {os.path.getsize(path) / 1e6:.1f} MB")
    print(f"{'='*60}")

    with h5py.File(path, "r") as f:
        print(f"\nTop-level keys: {list(f.keys())}")
        print(f"Attributes: {dict(f.attrs)}")

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name:40s} shape={obj.shape}  dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group:   {name:40s} keys={list(obj.keys())}  attrs={dict(obj.attrs)}")

        f.visititems(visit)


def main():
    data_dir = os.path.expandvars(sys.argv[1] if len(sys.argv) > 1 else "/scratch/$USER/data/active_matter")

    # Find HDF5 files
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(data_dir, "data", split)
        if not os.path.isdir(split_dir):
            split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"\n[{split}] Directory not found, tried: {split_dir}")
            continue

        files = sorted(glob.glob(os.path.join(split_dir, "*.hdf5")))
        print(f"\n[{split}] Found {len(files)} HDF5 files in {split_dir}")

        if files:
            # Inspect first file
            inspect_hdf5(files[0])

            # Show filename pattern
            print(f"\n  First 3 filenames:")
            for f in files[:3]:
                print(f"    {os.path.basename(f)}")


if __name__ == "__main__":
    main()
