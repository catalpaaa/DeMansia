""" Loader Factory, Fast Collate, CUDA Prefetcher
Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/loader.py and modified for token labeling
"""

import numpy as np
import torch.utils.data
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.distributed_sampler import OrderedDistributedSampler
from timm.data.random_erasing import RandomErasing

from .label_transforms_factory import create_token_label_transform
from .mixup import FastCollateTokenLabelMixup


def fast_collate(batch):
    """A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one tensor ordered by position
        # such that all tuple of position n will end up in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.int64)
        tensor = torch.zeros(
            (flattened_batch_size, *batch[0][0][0].shape), dtype=torch.uint8
        )
        for i in range(batch_size):
            assert (
                len(batch[i][0]) == inner_tuple_size
            )  # all input tensor tuples must be same length
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        if isinstance(batch[0][1], torch.Tensor):
            targets = torch.stack([b[1] for b in batch])
        else:
            targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False


class PrefetchLoader:

    def __init__(
        self,
        loader,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        fp16=False,
        re_prob=0.0,
        re_mode="const",
        re_count=1,
        re_num_splits=0,
    ):
        self.loader = loader
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
        self.fp16 = fp16
        if fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        if re_prob > 0.0:
            self.random_erasing = RandomErasing(
                probability=re_prob,
                mode=re_mode,
                max_count=re_count,
                num_splits=re_num_splits,
            )
        else:
            self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    next_input = next_input.half().sub_(self.mean).div_(self.std)
                else:
                    next_input = next_input.float().sub_(self.mean).div_(self.std)
                if self.random_erasing is not None:
                    next_input = self.random_erasing(next_input)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def mixup_enabled(self):
        if isinstance(self.loader.collate_fn, FastCollateTokenLabelMixup):
            return self.loader.collate_fn.mixup_enabled
        else:
            return False

    @mixup_enabled.setter
    def mixup_enabled(self, x):
        if isinstance(self.loader.collate_fn, FastCollateTokenLabelMixup):
            self.loader.collate_fn.mixup_enabled = x


def create_token_label_loader(
    dataset,
    input_size,
    batch_size,
    is_training=True,
    use_prefetcher=True,
    no_aug=False,
    re_prob=0.25,
    re_mode="pixel",
    re_count=1,
    re_split=False,
    scale=[0.08, 1.0],
    ratio=[3.0 / 4.0, 4.0 / 3.0],
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment='rand-m9-mstd0.5-inc1',
    num_aug_splits=0,
    interpolation="random",
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    num_workers=1,
    distributed=False,
    crop_pct=None,
    collate_fn=None,
    pin_memory=False,
    fp16=False,
    tf_preprocessing=False,
    use_multi_epochs_loader=False,
    use_token_label=True,
):
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    if use_token_label:
        transform_fn = create_token_label_transform
    else:
        transform_fn = create_transform
    dataset.transform = transform_fn(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )

    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)

    if collate_fn is None:
        collate_fn = (
            fast_collate
            if use_prefetcher
            else torch.utils.data.dataloader.default_collate
        )

    loader_class = torch.utils.data.DataLoader

    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader

    loader = loader_class(
        dataset,
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset)
        and sampler is None
        and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        persistent_workers=True,
    )
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.0
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            fp16=fp16,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
        )

    return loader


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)