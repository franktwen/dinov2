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

def _parse_slideflow_dataset(
    args: dict,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    import os
    os.environ["SF_BACKEND"] = "torch"
    import slideflow as sf
    from slideflow.io.torch import InterleaveIterator
    import torch.distributed as dist
    from torch.utils.data import IterableDataset
    
    # Check if distributed is initialized
    if dist.is_initialized():
        rank = dist.get_rank()
        num_replicas = dist.get_world_size()
        logger.info(f"DISTRIBUTED TRAINING DETECTED: rank={rank}, num_replicas={num_replicas}")
    else:
        rank = 0
        num_replicas = 1
        logger.warning("NO DISTRIBUTED TRAINING CONTEXT - both GPUs will read all data!")
        
    logger.info("Using slideflow args: \n{}".format(OmegaConf.to_yaml(args)))
    
    P = sf.load_project(args.project)
    dataset = P.dataset(
        tile_px=args.dataset.tile_px,
        tile_um=args.dataset.tile_um,
        **(dict() if 'dataset_kwargs' not in args else OmegaConf.to_container(args.dataset_kwargs))
    )
    tfrecords = dataset.tfrecords()
    labels = dataset.labels(args.outcome_labels)[0] if args.outcome_labels else None
    
    # Get distributed training info
    if dist.is_initialized():
        rank = dist.get_rank()
        num_replicas = dist.get_world_size()
    else:
        rank = 0
        num_replicas = 1
    
    interleave_kwargs = OmegaConf.to_container(args.interleave_kwargs) if args.interleave_kwargs else dict()
    interleave_kwargs.pop('seed', None)
    
    torch_dataset = InterleaveIterator(
        tfrecords,
        labels=labels,
        transform=None,
        standardize=False,
        use_labels=False,
        rank=rank,              # Are these actually being set?
        num_replicas=num_replicas,
        **interleave_kwargs
    )
    
    logger.info(f"Dataset created with rank={rank}, num_replicas={num_replicas}, num_tfrecords={len(tfrecords)}")
    
    class ApplyTransform(IterableDataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        def __iter__(self):
            for item in self.dataset:
                image = item[0]  # Extract image
                crops = self.transform(image)  # Get transformed dict
                yield (crops, 0)  # Wrap as tuple with dummy label
        def __len__(self):
            return len(self.dataset)

    return ApplyTransform(torch_dataset, transform)


def _wrap_slideflow_dataset(dataset, transform, target_transform):
    from torch.utils.data import IterableDataset
    import gc
    
    class SlideflowWrapper(IterableDataset):
        def __init__(self, dataset, transform, target_transform):
            self.dataset = dataset
            self.transform = transform
            self.target_transform = target_transform
            self._count = 0
        
        def __iter__(self):
            for item in self.dataset:
                self._count += 1
                
                # Extract image/label
                if isinstance(item, (tuple, list)):
                    image = item[0]
                    label = item[1] if len(item) > 1 else 0
                elif isinstance(item, dict):
                    image = item.get('image', item.get('img', None))
                    label = item.get('label', item.get('target', 0))
                else:
                    image = item
                    label = 0
                
                # Apply transforms
                if self.transform is not None:
                    image = self.transform(image)
                if self.target_transform is not None:
                    label = self.target_transform(label)
                
                yield image, label
                
                # Periodic garbage collection
                if self._count % 100 == 0:
                    gc.collect()
        
        def __len__(self):
            if hasattr(self.dataset, 'infinite') and self.dataset.infinite:
                return 10**9
            return len(self.dataset) if hasattr(self.dataset, '__len__') else 0
    
    return SlideflowWrapper(dataset, transform, target_transform)

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

    # ADD THIS CHECK:
    # IterableDatasets handle their own iteration, don't use a sampler
    from torch.utils.data import IterableDataset
    if isinstance(dataset, IterableDataset):
        sampler = None
        logger.info("dataset is IterableDataset, not using sampler")
    else:
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
