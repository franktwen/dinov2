"""WebDataset integration for DINOv2 training.

Drop-in replacement for the Slideflow dataset path in dinov2/data/loaders.py.
Reads pre-shuffled WebDataset tar shards sequentially — no random access,
no Slideflow dependency at training time.

Integration:
    1. Copy this file to dinov2/data/wds_loader.py
    2. In loaders.py, see integration instructions below.
"""

import glob
import io
import json
import logging
import os
from typing import Callable, Optional

import torch
import torch.distributed as dist
import webdataset as wds
from PIL import Image

logger = logging.getLogger("dinov2")

# Tissue label mapping (matches convert_to_webdataset.py)
TISSUE_LABELS = {
    "TCGA-BRCA": 0,
    "TCGA-COADREAD": 1,
    "TCGA-LUAD": 2,
    "TCGA-KIRC": 3,
}


class WebDatasetWrapper(torch.utils.data.IterableDataset):
    """WebDataset wrapper compatible with DINOv2's training loop.

    Wraps a WebDataset pipeline to provide:
    - Proper distributed sharding across GPUs
    - DINOv2-compatible (image, label) output format
    - Configurable epoch length
    - Shard shuffling between epochs

    Args:
        shard_dir: Directory containing tiles-*.tar shards and manifest.json
        transform: Image transform (DINOv2's DataAugmentationDINO)
        epoch_length: Number of batches per "epoch" (OFFICIAL_EPOCH_LENGTH)
        batch_size: Per-GPU batch size
        shuffle_buffer: Size of in-memory shuffle buffer
    """

    def __init__(
        self,
        shard_dir: str,
        transform: Optional[Callable] = None,
        epoch_length: int = 900,
        batch_size: int = 96,
        shuffle_buffer: int = 10000,
    ):
        super().__init__()
        self.shard_dir = shard_dir
        self.transform = transform
        self.epoch_length = epoch_length
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer

        # Find shards
        self.shards = sorted(glob.glob(os.path.join(shard_dir, "tiles-*.tar")))
        if not self.shards:
            raise ValueError(f"No tiles-*.tar shards found in {shard_dir}")

        # Load manifest
        manifest_path = os.path.join(shard_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            self._total_tiles = self.manifest.get("total_tiles", 0)
        else:
            self._total_tiles = len(self.shards) * 50000  # estimate
            self.manifest = {}

        logger.info(
            f"WebDataset: {len(self.shards)} shards, "
            f"{self._total_tiles:,} total tiles from {shard_dir}"
        )

    def _decode_sample(self, sample):
        """Decode a WebDataset sample into (image, label)."""
        jpeg_bytes = sample["jpg"]
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        meta = json.loads(sample["json"])
        tissue = meta.get("tissue", "unknown")
        label = TISSUE_LABELS.get(tissue, -1)

        return img, torch.tensor(label)

    def __iter__(self):
        shard_urls = [str(s) for s in self.shards]

        # Infinite iterator: re-create pipeline with fresh shard shuffle
        # each time all shards are exhausted.
        while True:
            pipeline = (
                wds.WebDataset(shard_urls, nodesplitter=wds.split_by_node, shardshuffle=True)
                .shuffle(self.shuffle_buffer)
                .map(self._decode_sample)
            )
            for sample in pipeline:
                yield sample

    def __len__(self):
        """Nominal length for DINOv2's progress tracking."""
        return self.epoch_length * self.batch_size


def make_webdataset(
    cfg_train: dict,
    image_transform: Optional[Callable] = None,
) -> WebDatasetWrapper:
    """Create a WebDataset from DINOv2 training config.

    Args:
        cfg_train: The cfg.train config node. Must contain:
            - slideflow.webdataset_path: path to shard directory
            - batch_size_per_gpu: batch size
            - OFFICIAL_EPOCH_LENGTH: batches per epoch
        image_transform: The DINOv2 DataAugmentationDINO transform.

    Returns:
        WebDatasetWrapper instance.
    """
    from omegaconf import OmegaConf

    # Extract config values
    if hasattr(cfg_train, 'slideflow'):
        sf_cfg = cfg_train.slideflow
    else:
        sf_cfg = cfg_train.get('slideflow', {})

    if hasattr(sf_cfg, 'webdataset_path'):
        wds_path = sf_cfg.webdataset_path
    else:
        wds_path = sf_cfg['webdataset_path']

    batch_size = getattr(cfg_train, 'batch_size_per_gpu', 96)
    epoch_length = getattr(cfg_train, 'OFFICIAL_EPOCH_LENGTH', 900)

    return WebDatasetWrapper(
        shard_dir=wds_path,
        transform=image_transform,
        epoch_length=epoch_length,
        batch_size=batch_size,
    )
