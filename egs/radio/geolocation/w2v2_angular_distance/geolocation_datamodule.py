from dataset import GeolocationDataset
from lhotse.utils import fix_random_seed

import argparse
from icefall.utils import str2bool
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (
    CutMix,
    DynamicBucketingSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class GeolocationDataModule:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="Geolocation data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--cuts",
            type=Path,
            default=Path("data/manifests/radio_cuts.jsonl.gz"),
            help="Path to cuts.",
        )
        group.add_argument(
            "--window-length",
            type=int,
            default=10,
            help="Cut chunks into windows of at most this duration. Default=10s"
        )
        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=False,
            help="When enabled, select noise from MUSAN and mix it "
            "with training dataset. ",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=100.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="The number of buckets for the BucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="When enabled, use on-the-fly cut mixing and feature "
            "extraction. Will drop existing precomputed feature manifests "
            "if available.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )
        group.add_argument(
            "--use-feats",
            type=str2bool,
            default=False,
            help="Whether to use features, such as FBANKs as input to the model"
        )
    
    def train_cuts(self) -> CutSet:
        logging.info("Getting Training Cuts")
        cuts = load_manifest_lazy(self.args.cuts)
        return cuts

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None, 
    ) -> DataLoader:

        logging.info("About to get Musan cuts")
        transforms = []
        if self.args.enable_musan:
            logging.info("Enable MUSAN")
            transforms.append(
                CutMix(cuts=cuts_musan, prob=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logging.info("Disable MUSAN")

        input_transforms = []
        
        logging.info("About to create train dataset")
        if self.args.on_the_fly_feats:
            train = GeolocationDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=64))),
                use_feats=self.args.use_feats,
            )
        else:
            train = GeolocationDataset(
                cut_transforms=transforms,
                input_transforms=input_transforms,
                use_feats=self.args.use_feats,
            )

        logging.info("Using DynamicBucketingSampler.")
        train_sampler = DynamicBucketingSampler(
            cuts_train,
            max_duration=self.args.max_duration,
            shuffle=False,
            num_buckets=self.args.num_buckets,
            drop_last=True,
            quadratic_duration=15,
        )
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(
        self,
        cuts_train: CutSet,
    ) -> DataLoader:

        logging.info("About to get Musan cuts")
        transforms = []
        input_transforms = []
        
        logging.info("About to create train dataset")
        if self.args.on_the_fly_feats:
            train = GeolocationDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=64))),
                use_feats=self.args.use_feats,
            )
        else:
            train = GeolocationDataset(
                cut_transforms=transforms,
                input_transforms=input_transforms,
                use_feats=self.args.use_feats,
            )

        logging.info("Using DynamicBucketingSampler.")
        train_sampler = DynamicBucketingSampler(
            cuts_train,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.info("About to create train dataloader")

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )

        return train_dl

