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

import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn

import lhotse
from lhotse import load_manifest_lazy
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed

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


from lhotse.utils import fix_random_seed
from lhotse.dataset import DynamicBucketingSampler
from torch.utils.data import DataLoader
from dataset import GeolocationDataset

import torch.nn.functional as F
import numpy as np

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]

lhotse.set_caching_enabled(True)

langs = [
    'af_za', 'am_et', 'ar_eg', 'as_in', 'ast_es', 'az_az',
    'be_by', 'bg_bg', 'bn_in', 'bs_ba', 'ca_es', 'ceb_ph',
    'ckb_iq', 'cmn_hans_cn', 'cs_cz', 'cy_gb', 'da_dk', 'de_de',
    'el_gr', 'en_us', 'es_419', 'et_ee', 'fa_ir', 'ff_sn',
    'fi_fi', 'fil_ph', 'fr_fr', 'ga_ie', 'gl_es', 'gu_in',
    'ha_ng', 'he_il', 'hi_in', 'hr_hr', 'hu_hu', 'hy_am',
    'id_id', 'ig_ng', 'is_is', 'it_it', 'ja_jp', 'jv_id',
    'ka_ge', 'kam_ke', 'kea_cv', 'kk_kz', 'km_kh', 'kn_in',
    'ko_kr', 'ky_kg', 'lb_lu', 'lg_ug', 'ln_cd', 'lo_la',
    'lt_lt', 'luo_ke', 'lv_lv', 'mi_nz', 'mk_mk', 'ml_in',
    'mn_mn', 'mr_in', 'ms_my', 'mt_mt', 'my_mm', 'nb_no',
    'ne_np', 'nl_nl', 'nso_za', 'ny_mw', 'oc_fr', 'om_et',
    'or_in', 'pa_in', 'pl_pl', 'ps_af', 'pt_br', 'ro_ro',
    'ru_ru', 'sd_in', 'sk_sk', 'sl_si', 'sn_zw', 'so_so',
    'sr_rs', 'sv_se', 'sw_ke', 'ta_in', 'te_in', 'tg_tj',
    'th_th', 'tr_tr', 'uk_ua', 'umb_ao', 'ur_pk', 'uz_uz',
    'vi_vn', 'wo_sn', 'xh_za', 'yo_ng', 'yue_hant_hk', 'zu_za',
]


lang2idx = {l: i for i, l in enumerate(langs)}


idx2lang = {i: l for i, l in enumerate(langs)}


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )  
    parser.add_argument(
        "--train-cuts",
        type=str,
    )
    parser.add_argument(
        "--valid-cuts",
        type=str,
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=400,
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=0,
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
        default=2000,
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

    parser.add_argument(
        "--narrowband",
        type=str2bool,
        default=False,
        help="Simulate narrowband",
    )

    add_model_arguments(parser)

    return parser


def add_model_arguments(parser):
    parser.add_argument("--freeze-iters", type=int, default=1000)
    parser.add_argument("--modelpath", type=str, default="facebook/mms-300m")
    parser.add_argument("--freeze-feat-extractor", type=str2bool, default=True)
    parser.add_argument("--pooling-type", type=str, default="pre")
    parser.add_argument("--cache-dir", type=str, default="/expscratch/mwiesner/testthis")


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50, #25
            "reset_interval": 200,
            "valid_interval": 500,  # 500
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
        cache_dir=params.cache_dir,
        freeze_feat_extractor=params.freeze_feat_extractor,
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
        return loss, info


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
        loss, loss_info  = compute_loss(
                params=params,
                model=model,
                batch=batch,
                is_training=False,
            )

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
            logging.info("UNFREEZING")
            unfreeze()
            for param_group in optimizer.param_groups:
                param_group['lr'] = init_lr
   
        with torch.cuda.amp.autocast(enabled=params.use_fp16):
            loss, loss_info = compute_loss(
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
        lr=params.lr,  # should have no effect
        betas=(0.9, 0.98), eps=1e-08, weight_decay=1e-06, 
    )
    #weight_decay=0.01, )
 
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

    train_cuts = load_manifest_lazy(params.train_cuts)
    train_cuts_dedup = []
    train_dups = set()
    for c in train_cuts:
        if c.id not in train_dups:
            train_cuts_dedup.append(c)
        train_dups.add(c.id)
    train_cuts = CutSet.from_cuts(train_cuts_dedup)
    
    valid_cuts = load_manifest_lazy(params.valid_cuts)
    valid_cuts_dedup = []
    valid_dups = set()
    for c in valid_cuts:
        if c.id not in valid_dups:
            valid_cuts_dedup.append(c)
        valid_dups.add(c.id)
    valid_cuts = CutSet.from_cuts(valid_cuts_dedup)
    
    train_cuts = train_cuts.filter(lambda c: c.duration < 30.0) # was 20.0 
    # Multiplex cuts.
    datasets = groupby(
        sorted(train_cuts.to_eager(), key=lambda x: x.supervisions[0].language),
        lambda x: x.supervisions[0].language
    )
    manifests = [CutSet.from_cuts(ds[1]).to_eager().shuffle() for ds in datasets] 
    total = sum(len(m) for m in manifests)
    train_cuts = CutSet.mux(
        *[c.repeat() for c in manifests],
        weights=[(len(m) / total)**0.7 for m in manifests],
    )
    valid_cuts = valid_cuts.shuffle()
    valid_cuts = valid_cuts.subset(first=1000)
    train_cuts = train_cuts.resample(16000)
    valid_cuts = valid_cuts.resample(16000)
    if args.narrowband:
        train_cuts = train_cuts.narrowband("mulaw")
        valid_cuts = valid_cuts.narrowband("mulaw")

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

    train_ds = GeolocationDataset()
    valid_ds = GeolocationDataset() 
    
    train_sampler = DynamicBucketingSampler(
        train_cuts,
        max_duration=params.max_duration,
        shuffle=True,
        num_buckets=params.num_buckets,
        drop_last=True,
        quadratic_duration=15,
    )
    valid_sampler = DynamicBucketingSampler(
        valid_cuts,
        max_duration=params.max_duration,
        shuffle=False,
    )
    
    seed = torch.randint(0, 100000, ()).item()
    worker_init_fn = _SeedWorkers(seed)
    
    train_dl = DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=None,
        num_workers=params.num_workers,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
    )
    valid_dl = DataLoader(
        valid_ds,
        sampler=valid_sampler,
        batch_size=None,
        num_workers=params.num_workers,
        persistent_workers=False,
    )

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    epoch = 1
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

        #save_checkpoint(
        #    params=params,
        #    model=model,
        #    model_avg=model_avg,
        #    optimizer=optimizer,
        #    scheduler=scheduler,
        #    sampler=train_dl.sampler,
        #    scaler=scaler,
        #    rank=rank,
        #)

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()




def main():
    parser = get_parser()
    args = parser.parse_args()
    Path(args.exp_dir).mkdir(mode=511, parents=True, exist_ok=True)
    with open(str(args.exp_dir) + "/args.conf", 'w', encoding='utf-8') as f:
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
lhotse.set_audio_duration_mismatch_tolerance(0.1)

if __name__ == "__main__":
    main()

