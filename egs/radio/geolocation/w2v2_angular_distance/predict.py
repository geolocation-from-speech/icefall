#!/usr/bin/env python3
# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo,
#                                                       Zengwei Yao,
#                                                       Daniel Povey)
# Copyright    2023       Johns Hopkins University (authors: Matthew Wiesner,
#                                                            Patrick Foley)
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    Usage:

        python w2v2_angular_distance/predict.py \
          --world-size ${ngpu} \
          --max-duration ${max_duration} \
          --exp-dir "${exp_name}" \
          --num-buckets 30 \
          --num-workers 8 \
          --use-feats False \
          --use-fp16 True \
          --cuts data/manifests/radio_cuts.jsonl.gz \
"""

from geolocation_datamodule import GeolocationDataModule
import argparse
import copy
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union
from itertools import chain, groupby
import math
from tqdm import tqdm
import re

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from lhotse.cut import Cut
from lhotse import CutSet
from lhotse.dataset.sampling.base import CutSampler
from lhotse.dataset.unsupervised import RecordingChunkIterableDataset
from lhotse.utils import fix_random_seed
from lhotse import load_manifest_lazy

from train import angular_distance
from model import Wav2Vec2Model
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    setup_logger,
    str2bool,
)

import torch.nn.functional as F
import numpy as np


valid_stations="""m4VESYpw
SiJV_lYK
SYCyZWnI
uztnKK23
MUq5WAJ8
AtjC0lVm
hFtZdlD7
D9W8VNEe
ThwOsInM
HHikF5mO
KQF445vh
5awguDj0
AnP3NyFu
uZeYNZzC
rrGxSGIb
vwwkadDl
5TGMtep5
puYSMon7
vcZg0CDI
rI3loVta
tl357Kz5
7Iz7yw5K
Mzcoa0KE
dvh1nQq1
Fz_77-J9
BA0Oh4Qu
wAsFnJLS
xxbjL4cw
I2C6Ww3m
BzAIOYkL
SMuks16K
Rj35oQRS
Kep0jAq_
Q-cyIHST
d9zbZG28
GY2CPOWQ
CQAAS1K7
FirzXrVe
CtPgWJda
gPGeP1Ly
CP4KHede
EsGd5eWD
5OtZR0vB
vjUza7a5
9urNOQkl
pfPYAQRu
GGxK0ewe
mHrRIfhQ
RD_IF7Lp
XybUjPHm"""


valid_stations = valid_stations.split("\n")


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--valid-cuts",
        type=str,
        default="data/manifests/radio_valid_cuts.jsonl.gz"
    ) 
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
        default="checkpoint-100000",
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
        "--average-over-segments",
        type=str2bool,
        default=False,
        help="After decoding segments, the flag decides whether segments "
        "matched by the segment_average_pattern (see option) should be "
        "to predict a final distance. In the case of the radio data, e.g., "
        "all segments from the 30 sec recording can be averaged to compute the "
        "estimated location for the recording, enabling a fair comparison "
        "among systems that use different window sizes."
    )

    parser.add_argument(
        "--segment-average-pattern",
        type=str,
        default=".*lat[^_]+_long[^_]+",
        help="The regex pattern used to match which segment ids should be "
        "grouped together when computing a distance when the "
        "average_over_segments flag is True. Note that backslashes need to be "
        "escaped as the pattern is passed to an f-string and not a raw string."
    )

    parser.add_argument("--suffix", type=str, default="")

    return parser


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
            "env_info": get_env_info(),
        }
    )
    return params


def get_model(params: AttributeDict) -> nn.Module:
    with open(Path(params.exp_dir) / "args.conf", 'r') as f:
        conf = eval(f.readline())
    
    model = Wav2Vec2Model(
        modelpath=conf['modelpath'],
        pooling_loc=conf['pooling_loc'],
        pooling_type=conf['pooling_type'],
    ) 
    return model


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    y_polar = batch["supervisions"]["targets"] * (math.pi / 180)
    y_polar = y_polar.to(device)
    feature = batch["inputs"].to(device)
    feature_lens = batch["features_lens"].to(device)
    with torch.set_grad_enabled(is_training):
        x, w = model(feature, feature_lens)
        # Convert targets to cartesian coordinates
        cos_lat = torch.cos(y_polar[:, 0]).unsqueeze(-1)
        cos_lon = torch.cos(y_polar[:, 1]).unsqueeze(-1)
        sin_lat = torch.sin(y_polar[:, 0]).unsqueeze(-1)
        sin_lon = torch.sin(y_polar[:, 1]).unsqueeze(-1)
        y_cartesian = torch.cat(
            [
                cos_lat * cos_lon,
                cos_lat * sin_lon,
                sin_lat
            ], dim=1
        ).to(device)

        # If regression x will be B x 3, where the dims are
        # cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat)
        pred_lon = torch.atan2(x[:, 0], x[:, 1]).unsqueeze(-1) 
        pred_lat = torch.asin(x[:, 2]).unsqueeze(-1)
        x_polar = torch.cat((pred_lat, pred_lon), dim=1).to(device) 
        x_cartesian = x

        l = angular_distance(y_polar, x_polar)
        loss = l.sum()
        info = MetricsTracker()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
   
        info["loss"] = loss.detach().cpu().item()
        info[f"avg_dist"] = 6378.1 * loss.detach().cpu().item()
        pred = x_polar.mul(180. / math.pi).cpu().detach()
        info["frames"] = x.size(0)
        return loss, info, y_polar * (180 / math.pi), pred


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
    logging.info("Prediction started")
    
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

    radio_ssl = GeolocationDataModule(args)
    cuts = load_manifest_lazy(args.valid_cuts)
    valid_dl = radio_ssl.valid_dataloaders(cuts)

    valid_info, preds, tgts, ids = predict(
        params=params,
        model=model,
        model_avg=model_avg,
        valid_dl=valid_dl,
        world_size=world_size,
        rank=rank,
    )

    tgts = torch.cat(tgts, dim=0)
    preds = torch.cat(preds, dim=0)
    logging.info(valid_info)
    np.save(f'{params.exp_dir}/results/tgts_{params.suffix}.npy', tgts.cpu().numpy())
    np.save(f'{params.exp_dir}/results/preds_{params.suffix}.npy', preds.numpy())
    np.save(f'{params.exp_dir}/results/ids_{params.suffix}.npy', np.array(ids))
    if params.average_over_segments:
        logging.info("Averaging over segments ...")
        X, X_, Y = [], [], []
        for i, j  in groupby(zip(ids, preds, tgts), lambda x: re.search(f"{params.segment_average_pattern}", x[0]).group(0)):
            j_ = list(j)
            preds_ = []
            preds = []
            for x in j_:
                preds.append(x[1].unsqueeze(0).cpu() * (math.pi / 180))
                pred = x[1].unsqueeze(0).cpu() * (math.pi / 180)
                tgt = x[2].unsqueeze(0).cpu()
                cos_lat = torch.cos(pred[:, 0]).unsqueeze(-1)
                cos_lon = torch.cos(pred[:, 1]).unsqueeze(-1)
                sin_lat = torch.sin(pred[:, 0]).unsqueeze(-1)
                sin_lon = torch.sin(pred[:, 1]).unsqueeze(-1)
                pred_cartesian = torch.cat(
                    [
                        cos_lat * cos_lon,
                        cos_lat * sin_lon,
                        sin_lat
                    ], dim=1
                ).to(device)
                preds_.append(pred_cartesian) 
            pred_ = sum(preds_) / len(preds_)
            pred = sum(preds) / len(preds) 
            pred_ = pred_ / pred_.norm()
            pred_lon = torch.atan2(pred_[:, 1], pred_[:, 0]).unsqueeze(-1) 
            pred_lat = torch.asin(pred_[:, 2]).unsqueeze(-1)
            pred_polar = torch.cat((pred_lat, pred_lon), dim=1).to(device) 
            X_.append(pred_polar)  
            Y.append(tgt)
            X.append(pred)
        X_ = torch.cat(X_, dim=0).cpu()
        X = torch.cat(X, dim=0).cpu()
        Y = torch.cat(Y, dim=0).cpu()
        avg_dist_ = 6378.1 * angular_distance(Y * (math.pi / 180), X_) / X_.size(0)
        avg_dist = 6378.1 * angular_distance(Y * (math.pi / 180), X) / X.size(0)
        with open(params.exp_dir / "results" / f"results_{params.suffix}.txt", 'w') as f:
            print(valid_info, file=f)
            print(f"seg_averaged_distance: {avg_dist_.sum().item()}", file=f)
        logging.info(f"seg_averaged_distance: {avg_dist_.sum().item()}")
        logging.info(f"seg_averaged_distance naive: {avg_dist.sum().item()}")
    else:
        with open(params.exp_dir / "results" / f"results_{params.suffix}.txt", 'w') as f:
            print(valid_info, file=f)

    logging.info("Done!")
    
    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    GeolocationDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    results_dir = args.exp_dir / "results"
    results_dir.mkdir(mode=511, parents=True, exist_ok=True)
    with open(str(results_dir) + f"/args_{args.suffix}.conf", 'w', encoding='utf-8') as f:
        to_print = {k: v for k, v in vars(args).items()}
        to_print['exp_dir'] = str(to_print['exp_dir'])
        to_print['cuts'] = str(to_print['cuts'])
        to_print['valid_cuts'] = str(to_print['valid_cuts']) 
        print(to_print, file=f)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)



if __name__ == "__main__":
    main()
