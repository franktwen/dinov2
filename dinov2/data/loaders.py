# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar
from omegaconf import OmegaConf

import torch
from torch.utils.data import Sampler

from .datasets import ImageNet, ImageNet22k
from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler


logger = logging.getLogger("dinov2")


class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


def _make_bool_str(b: bool) -> str:
    return "yes" if b else "no"


def _make_sample_transform(image_transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    def transform(sample):
        image, target = sample
        if image_transform is not None:
            image = image_transform(image)
        if target_transform is not None:
            target = target_transform(target)
        return image, target

    return transform


def _parse_dataset_str(dataset_str: str):
    tokens = dataset_str.split(":")

    name = tokens[0]
    kwargs = {}

    for token in tokens[1:]:
        key, value = token.split("=")
        assert key in ("root", "extra", "split")
        kwargs[key] = value

    if name == "ImageNet":
        class_ = ImageNet
        if "split" in kwargs:
            kwargs["split"] = ImageNet.Split[kwargs["split"]]
    elif name == "ImageNet22k":
        class_ = ImageNet22k
    else:
        raise ValueError(f'Unsupported dataset "{name}"')

    return class_, kwargs

# In dinov2/data/loaders.py

def _parse_slideflow_dataset(
    args: dict,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    import os
    os.environ["SF_BACKEND"] = "torch"
    import slideflow as sf
    from slideflow.io.torch import IndexedInterleaver
    import torch.distributed as dist
    import time
    import hashlib # For hashing

    if dist.is_initialized():
        rank = dist.get_rank()
        num_replicas = dist.get_world_size()
    else:
        rank = 0
        num_replicas = 1

    logger.info("Using slideflow args: \n{}".format(
        OmegaConf.to_yaml(args)
    ))

    # --- START OF AUTOMATED CACHING LOGIC ---

    # Step 1: Get the list of TFRecord files first. This is determined by your filters.
    P = sf.load_project(args.project)
    dataset_obj = P.dataset(
        tile_px=args.dataset.tile_px,
        tile_um=args.dataset.tile_um,
        **(dict() if 'dataset_kwargs' not in args else OmegaConf.to_container(args.dataset_kwargs))
    )
    tfrecords = dataset_obj.tfrecords()

    # Step 2: Create a unique hash from the list of files.
    # This ensures if the filters change (and thus the tfrecords list changes), a new cache is made.
    tfrecord_string = "".join(sorted(tfrecords))
    config_hash = hashlib.md5(tfrecord_string.encode()).hexdigest()[:10] # A 10-char hash is enough

    # Step 3: Define the cache path using this unique hash.
    cache_filename = f'cached_indices_{config_hash}_rank_{rank}.pt'
    cache_path = os.path.join(args.project, cache_filename)

    if os.path.exists(cache_path):
        logger.info(f"[Rank {rank}] Found VALID cached index ({cache_filename}). Loading...")
        start_time = time.time()
        cached_data = torch.load(cache_path)
        labels = cached_data['labels']
        indices = cached_data['indices']
        logger.info(f"[Rank {rank}] Loaded cached index in {time.time() - start_time:.2f} seconds.")
    else:
        logger.info(f"[Rank {rank}] No valid cache found for this config. Building from scratch...")
        start_time = time.time()
        if args.outcome_labels:
            labels = dataset_obj.labels(args.outcome_labels)[0]
        else:
            labels = None

        logger.info(f"[Rank {rank}] Calling IndexedInterleaver to build index...")
        temp_dataset = IndexedInterleaver(tfrecords, rank=rank, num_replicas=num_replicas)
        indices = temp_dataset.indices

        logger.info(f"[Rank {rank}] Saving index to new cache file: {cache_filename}")
        torch.save({'labels': labels, 'indices': indices}, cache_path)
        logger.info(f"[Rank {rank}] Index built and cached in {time.time() - start_time:.2f} seconds.")

    if dist.is_initialized():
        dist.barrier()
    # --- END OF AUTOMATED CACHING LOGIC ---

    interleave_kwargs = OmegaConf.to_container(args.interleave_kwargs) if args.interleave_kwargs else dict()

    torch_dataset = IndexedInterleaver(
        tfrecords,
        indices=indices,
        labels=labels,
        transform=transform,
        standardize=False,
        rank=rank,
        num_replicas=num_replicas,
        **interleave_kwargs
    )
    return torch_dataset

def make_dataset(
    *,
    dataset_str: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    slideflow_args: Optional[dict] = None,
):
    """
    Creates a dataset with the specified parameters.

    Args:
        dataset_str: A dataset string description (e.g. ImageNet:split=TRAIN).
        transform: A transform to apply to images.
        target_transform: A transform to apply to targets.

    Returns:
        The created dataset.
    """
    logger.info(f'using dataset: "{dataset_str}"')

    if slideflow_args is not None:
        dataset = _parse_slideflow_dataset(slideflow_args, transform, target_transform)
    else:
        class_, kwargs = _parse_dataset_str(dataset_str)
        dataset = class_(transform=transform, target_transform=target_transform, **kwargs)

    logger.info(f"# of dataset samples: {len(dataset):,d}")

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)
    if not hasattr(dataset, "target_transform"):
        setattr(dataset, "target_transform", target_transform)

    return dataset


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        # TODO: Remove support for old shuffling
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, SHARDED_INFINITE_NEW, DISTRIBUTED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        sampler_advance: How many samples to skip (when applicable).
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
    """

    sampler = _make_sampler(
        dataset=dataset,
        type=sampler_type,
        shuffle=shuffle,
        seed=seed,
        size=sampler_size,
        advance=sampler_advance,
    )

    logger.info("using PyTorch data loader")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader
