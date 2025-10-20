import argparse
import copy
import logging
logging.basicConfig(level=logging.INFO)
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union
from itertools import chain, groupby
import json
import math
from tqdm import tqdm
import re

import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn

import lhotse
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from lhotse import load_manifest_lazy

from lhotse import CutSet

from model_mean import LIDModel
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam


from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
)

from lhotse.dataset import DynamicBucketingSampler
from torch.utils.data import DataLoader
from dataset import GeolocationDataset

import torch.nn.functional as F
import numpy as np


langs = [
    "afr-afr",
    "ara-aeb",
    "ara-arq",
    "ara-ayl",
    "eng-ens",
    "eng-iaf",
    "fra-ntf",
    "nbl-nbl",
    "orm-orm",
    "tir-tir",
    "tso-tso",
    "ven-ven",
    "xho-xho",
    "zul-zul",
]


lang2idx = {l: i for i, l in enumerate(langs)}


idx2lang = {i: l for i, l in enumerate(langs)}


LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


lhotse.set_caching_enabled(True)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )  
    parser.add_argument("--cuts", type=str)
    parser.add_argument("--max-duration", type=float, default=800)
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint-32000",
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/expscratch/mwiesner/testthis"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )

    parser.add_argument("--suffix", type=str, default="")
    add_model_arguments(parser)

    return parser


def add_model_arguments(parser):
    parser.add_argument("--modelpath", type=str, default="facebook/mms-300m")
    parser.add_argument("--pooling-type", type=str, default="pre")


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 1, #25
            "reset_interval": 200,
            "valid_interval": 10,  # 500
            # parameters for zipformer
            "feature_dim": 64,
            "env_info": get_env_info(),
        }
    )
    return params


def get_model(params: AttributeDict) -> nn.Module:
    model = LIDModel(
        len(lang2idx),
        pooling_type=params.pooling_type,
        modelpath=params.modelpath,
        cache_dir=params.cache_dir
    )
    return model


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params



def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    langs = batch["supervisions"]["language"]
    y = torch.LongTensor([lang2idx[l] for l in langs]).to(device) 
    feature = batch["inputs"].to(device)
    feature_lens = batch["features_lens"].to(device)
    with torch.set_grad_enabled(is_training):
        x, w = model(feature, feature_lens)
        
        l = F.cross_entropy(x, y, reduction='sum')
        correct = sum(x.argmax(-1) == y)
        loss = l.sum()
        info = MetricsTracker()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
   
        info["loss"] = loss.detach().cpu().item()
        info["acc"] = correct
        info["frames"] = x.size(0)
        return loss, info, langs, x.cpu()


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    ids, tgts, preds = [], [], []
    for batch_idx, batch in tqdm(enumerate(valid_dl)):
        loss, loss_info, tgts_, preds_  = compute_loss(
                params=params,
                model=model,
                batch=batch,
                is_training=False,
            )
        ids_ = batch['ids']
        ids.extend(ids_)
        tgts.append(tgts_)
        preds.append(preds_)

        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    return tot_loss, preds, tgts, ids


def predict(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    model_avg: Optional[nn.Module] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    
    valid_info, preds, tgts, ids = compute_validation_loss(
        params=params,
        model=model,
        valid_dl=valid_dl,
        world_size=world_size,
    )
    return valid_info, preds, tgts, ids 
   
    
def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")
    
    tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    logging.info(params)

    logging.info("About to create model")
    
    model = get_model(params)
    
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    model_avg: Optional[nn.Module] = None
   
    mdl = torch.load(f"{params.exp_dir}/{params.checkpoint}.pt", map_location="cpu")
    
    model.load_state_dict(mdl['model'])

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if params.inf_check:
        register_inf_check_hooks(model)

    valid_cuts = load_manifest_lazy(args.cuts)
    valid_cuts = valid_cuts.resample(16000)
    valid_cuts_dedup = []
    valid_dups = set()
    for c in valid_cuts:
        if c.id not in valid_dups:
            valid_cuts_dedup.append(c)
        valid_dups.add(c.id)
    valid_cuts = CutSet.from_cuts(valid_cuts_dedup)

    
    valid_ds = GeolocationDataset() 

    valid_sampler = DynamicBucketingSampler(
        valid_cuts,
        max_duration=params.max_duration,
        shuffle=False,
    )

    valid_dl = DataLoader(
        valid_ds,
        sampler=valid_sampler,
        batch_size=None,
        num_workers=params.num_workers,
        persistent_workers=False,
    )
    #if not params.print_diagnostics:
    #    scan_pessimistic_batches_for_oom(
    #        model=model,
    #        train_dl=train_dl,
    #        optimizer=optimizer,
    #        params=params,
    #    )

    
    valid_info, preds, tgts, ids = predict(
        params=params,
        model=model,
        model_avg=model_avg,
        valid_dl=valid_dl,
        world_size=world_size,
        rank=rank,
    )

    #tgts = torch.cat(tgts, dim=0)
    tgts = np.concatenate(tgts)
    preds = torch.cat(preds, dim=0)
    print(valid_info)
    if world_size > 1:
        with open(params.exp_dir / "results" / f"results_{params.suffix}_{rank}.txt", 'w') as f:
            print(valid_info, file=f)
        #np.save(f'{params.exp_dir}/results/tgts_{params.suffix}.npy', tgts.cpu().numpy())
        np.save(f'{params.exp_dir}/results/tgts_{params.suffix}_{rank}.npy', tgts)
        np.save(f'{params.exp_dir}/results/scores_{params.suffix}_{rank}.npy', preds.numpy())
        np.save(f'{params.exp_dir}/results/ids_{params.suffix}_{rank}.npy', np.array(ids))

    else:
        with open(params.exp_dir / "results" / f"results_{params.suffix}.txt", 'w') as f:
            print(valid_info, file=f)
        #np.save(f'{params.exp_dir}/results/tgts_{params.suffix}.npy', tgts.cpu().numpy())
        np.save(f'{params.exp_dir}/results/tgts_{params.suffix}.npy', tgts)
        np.save(f'{params.exp_dir}/results/scores_{params.suffix}.npy', preds.numpy())
        np.save(f'{params.exp_dir}/results/ids_{params.suffix}.npy', np.array(ids))
    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()




def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    results_dir = args.exp_dir / "results"
    results_dir.mkdir(mode=511, parents=True, exist_ok=True)
    with open(str(results_dir) + f"/args_{args.suffix}.conf", 'w', encoding='utf-8') as f:
        print(vars(args), file=f)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
lhotse.set_audio_duration_mismatch_tolerance(0.1)

if __name__ == "__main__":
    main()


# This code is stuff I don't want to forget that was used for the contrastive
# loss
#      #numerators = F.cosine_similarity(x, cartesian_targets)
            #if params.num_negs > 0:
            #    negs = make_negs(params.num_negs).to(device)
            #    targets = torch.cat([cartesian_targets, negs], dim=0)
            #    denominators = F.cosine_similarity(
            #        x.repeat_interleave(targets.size(0), dim=0),
            #        targets.repeat(x.size(0), 1)
            #    ).view(x.size(0), -1).logsumexp(dim=1)
            #    losses.append((numerators - denominators).sum())
            #else:
            #    losses.append((0.5 * (numerators - 1.0)).sum()) # 0.5*(x -1) makes the range (-1, 0)
            
            ##if x.size(0) > 1:
            ##    denominators_ = F.cosine_similarity(
            ##        x.repeat(x.size(0), 1),
            ##        cartesian_targets.repeat_interleave(x.size(0), dim=0)
            ##    ).view(x.size(0), -1).logsumexp(dim=1)
            ##    losses.append((numerators - denominators_).sum())

