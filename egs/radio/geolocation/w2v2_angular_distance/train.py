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

        python w2v2_angular_distance/train.py \
          --world-size ${ngpu} \
          --max-duration ${max_duration} \
          --exp-dir "${exp_name}" \
          --num-buckets 30 \
          --num-workers 8 \
          --use-feats False \
          --use-fp16 True \
          --lr 3e-06 \
          --pct-start 0.08 \
          --num-epochs 70 \
          --total-steps 800000 \
          --freeze-iters 1000 \
          --freeze-lr 0.00001 \
          --modelpath facebook/mms-300m \
          --pooling-type att \
          --pooling-loc 0 \
          --freeze-feat-extractor True \
          --cuts data/manifests/radio_cuts.jsonl.gz
"""
from geolocation_datamodule import GeolocationDataModule
import argparse
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union
import math

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from lhotse import load_manifest_lazy

from model import Wav2Vec2Model
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
    setup_logger,
    str2bool,
)


LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler]


# This list of stations were used for validation data
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
        "--debug",
        type=str2bool,
        default=False,
        help="This is useful for modifying the training / model to be faster "
        "to load and run when debugging. The amount of training data is reduced."
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
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
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
        "--lr", type=float, default=5e-04, help="The base learning rate."
    )

    parser.add_argument(
        "--freeze-lr", type=float, default=0.01,
        help="The learning rate used to train new unfrozen parameters that are "
        "introduced on top of the base wav2vec2 model. We normally first train "
        "these parameters for some number of iterations/"
    )

    parser.add_argument(
        "--pct-start", type=float, default=0.08, help="The percent of steps "
        "to use as warmup",
    )

    parser.add_argument(
        "--total-steps", type=int, default=400000, help="The total number of "
        "steps for the lr_scheduler",
    )    
    
    parser.add_argument(
        "--weight-decay", type=float, default=1e-06, help="Adam weight decay",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=10000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 1.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    add_model_arguments(parser)

    return parser


def add_model_arguments(parser):
    parser.add_argument("--freeze-iters", type=int, default=1000,
        help="The number of iterations the base wav2vec2 model is frozen."
    )
    parser.add_argument("--modelpath", type=str, default="facebook/mms-300m",
        help="The huggingface path for the base wav2vec2 model."
    )
    parser.add_argument("--pooling-type", type=str, default="att",
        choices=["att", "avg"], help="The pooling mechanism to use."
    )
    parser.add_argument("--pooling-heads", type=int, default=1)
    parser.add_argument("--pooling-loc", type=int, default=0, choices=[0, 1, 2],
        help="An integer specifying where in the model representations should "
        "be pooled.\n"
        "0 --> after wav2vec2,\n"
        "1 --> after projection to cartesian coordinates\n"
        "2 --> after projction onto sphere representing the Earth"
    )
    parser.add_argument("--freeze-feat-extractor", type=str2bool, default=True,
        help="Specifies whether to freeze the wav2vec2.0 feature extractor."
    )


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 100, #25
            "reset_interval": 200,
            "valid_interval": 1000,  # 500
            # parameters for zipformer
            "feature_dim": 64,
            "env_info": get_env_info(),
        }
    )
    return params


def get_model(params: AttributeDict) -> nn.Module:
    model = Wav2Vec2Model(
        modelpath=params.modelpath,
        freeze_feat_extractor=params.freeze_feat_extractor,
        pooling_loc=params.pooling_loc,
        pooling_type=params.pooling_type,
        pooling_heads=params.pooling_heads,
    )
    return model


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)


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


def angular_distance(Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    '''
        angular distance points are (lat, long). Rows are points, cols are
        lat, lon. Values are assumed to be in radian.

        :param X: The tensor of hypothesis locations (lat,lon coordinates)
        :param Y: The tensor of target locations (lat,lon coordinates)
        :return: The angular distance between each element in X and the
            corresponding element in Y.
    '''
    h = (
        torch.sin(X[:, 0])*torch.sin(Y[:, 0]) + 
        torch.cos(X[:, 0])*torch.cos(Y[:, 0])*torch.cos(X[:, 1] - Y[:, 1])
    )
    return torch.acos(h.clamp(-1, 1))


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker, Tensor, Tensor]:
    """
        Compute the angular distance between the hypothesized and target
        locations. Convert cartesian outputs to spherical coordinates in rads.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    # Convert to radians
    y_polar = batch["supervisions"]["targets"] * (math.pi / 180)
    y_polar = y_polar.to(device)
    feature = batch["inputs"].to(device)
    feature_lens = batch["features_lens"].to(device)
    
    # Get cartesian predictions
    with torch.set_grad_enabled(is_training):
        x, w = model(feature, feature_lens)
         
        # For regression x will be B x 3, where the dims are
        # x = cos(lat)*cos(lon), y = cos(lat)*sin(lon), z = sin(lat)
        # lat = arctan(y / x), lon = arcsin(z)
        pred_lon = torch.atan2(x[:, 0], x[:, 1]).unsqueeze(-1) 
        pred_lat = torch.asin(x[:, 2]).unsqueeze(-1)
        x_polar = torch.cat((pred_lat, pred_lon), dim=1).to(device) 

        l = angular_distance(y_polar, x_polar)
        loss = l.sum()
        info = MetricsTracker()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
  
        # 6378.1 km is just the equatorial radius, so this gives an upper bound
        # on the distance away that points might actually be.
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

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info, tgts, preds  = compute_loss(
                params=params,
                model=model,
                batch=batch,
                is_training=False,
            )

        if batch_idx % 10 == 0:
            logging.info(f"tgt: {tgts[0].tolist()}")
            logging.info(f"hyp: {preds[0].tolist()}")
        
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss



def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    init_lr: float,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    model.train()
    tot_loss = MetricsTracker()
    
    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint_impl(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    for batch_idx, batch in enumerate(train_dl): 
        params.batch_idx_train += 1
        batch_size = batch["inputs"].size(0)
        is_frozen = model.module.frozen if isinstance(model, DDP) else model.frozen
        unfreeze = model.module.unfreeze_encoder if isinstance(model, DDP) else model.unfreeze_encoder
        if params.batch_idx_train > params.freeze_iters and is_frozen:
            logging.info("Unfreezing ...")
            unfreeze()
            for param_group in optimizer.param_groups:
                param_group['lr'] = init_lr
            
        with torch.autocast("cuda", enabled=params.use_fp16):
            loss, loss_info, tgts, preds = compute_loss(
                params=params,
                model=model,
                batch=batch,
                is_training=True,
            )
        # summary stats
        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

        # NOTE: We use reduction==sum and loss is computed over utterances
        # in the batch and there is no normalization to it so far.
        scaler.scale(loss).backward()
        if not is_frozen:
            scheduler.step()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
            and model_avg is not None
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if batch_idx % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 8.0 or (cur_grad_scale < 32.0 and batch_idx % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise RuntimeError(
                    f"grad_scale is too small, exiting: {cur_grad_scale}"
                )

        if batch_idx % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr()) if not is_frozen else params.freeze_lr
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0
            
            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )
            logging.info(f"tgt: {tgts[0, :2]}")
            logging.info(f"hyp: {preds[0, :2]}")
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        if batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            logging.info("---------------------------------------------------")
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info("---------------------------------------------------")
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


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
    
    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
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

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
   
    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Use Hubert Optimizer and Scheduler here
    optimizer = Adam(
        list(filter(lambda p: p.requires_grad, model.parameters())),
        lr=params.lr,
        betas=(0.9, 0.98), eps=1e-08, weight_decay=params.weight_decay, 
    )
 
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=params.lr,
        total_steps=params.total_steps,
        pct_start=params.pct_start,
        anneal_strategy="cos",
        div_factor=200,
    )

    init_lr = scheduler.get_lr()[0]
    if params.batch_idx_train < params.freeze_iters:
        if isinstance(model, DDP):
            model.module.freeze_encoder()
        else:
            model.freeze_encoder()
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.freeze_lr 

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            2**22
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    if args.debug:
        args.num_buckets = 3

    radio_ssl = GeolocationDataModule(args)

    cuts = radio_ssl.train_cuts()
    # Remove any really short segments
    cuts = cuts.filter(lambda c: c.duration >= 2.0)
    train_cuts = cuts.filter(lambda c: c.supervisions[0].custom['station'] not in valid_stations)
    #valid_cuts = cuts.filter(lambda c: c.supervisions[0].custom['station'] in valid_stations)
    #valid_cuts = valid_cuts.subset(first=2000)
    valid_cuts = load_manifest_lazy("data/manifests/fleurs_dev/cuts_11.jsonl.gz")
    
    if args.debug:
        debug_cuts = train_cuts.shuffle().subset(first=50)
        valid_cuts = debug_cuts 
        params.log_interval = 1
    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    if args.debug:
        train_dl = radio_ssl.train_dataloaders(
            debug_cuts, sampler_state_dict=sampler_state_dict
        )
    else:
        train_dl = radio_ssl.train_dataloaders(
            train_cuts, sampler_state_dict=sampler_state_dict
        )

    valid_dl = radio_ssl.valid_dataloaders(valid_cuts)

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch
        
        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
            init_lr=init_lr
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    GeolocationDataModule.add_arguments(parser)
    args = parser.parse_args()
    Path(args.exp_dir).mkdir(mode=511, parents=True, exist_ok=True)
    with open(str(args.exp_dir) + "/args.conf", 'w', encoding='utf-8') as f:
        to_print = vars(args)
        to_print['exp_dir'] = str(to_print['exp_dir']) 
        to_print['cuts'] = str(to_print['cuts'])
        print(vars(args), file=f)

    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)



torch.set_num_threads(1)
torch.set_num_interop_threads(1)


if __name__ == "__main__":
    main()
