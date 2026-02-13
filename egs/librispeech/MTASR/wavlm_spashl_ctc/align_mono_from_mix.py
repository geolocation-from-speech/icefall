#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang
#                                                  Mingshuang Luo)
#
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
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
  ./zipformer_ctc/train.py \
     --exp-dir ./zipformer_ctc/exp \
     --world-size 4 \
     --full-libri 1 \
     --max-duration 500 \
     --num-epochs 30
"""

import argparse
import copy
import logging
import math
import json

from collections import defaultdict
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union, List

import os
import time
from tqdm import tqdm
import torch
import k2
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from lhotse import CutSet
from model import MDCTCModel
from torch.optim import Adam
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from icefall import diagnostics
from mdctc_graph_compiler import MDCTCGraphCompiler
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.err import raise_grad_scale_is_too_small_error
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
)

from lhotse.utils import compute_num_frames
import numpy as np
import torchaudio


LRSchedulerType = torch.optim.lr_scheduler._LRScheduler


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=55,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    
    parser.add_argument(
        "--suffix",
        type=str,
        help="decoding suffix."
    )

    parser.add_argument(
        "--blank-weight",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer_mdctc/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_5000/bpe.model",
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_bpe_5000",
        help="""The lang dir
        It contains language related input files such as
        "lexicon.txt"
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--frame-duration",
        type=float,
        default=0.04,
    )

    parser.add_argument(
        "--collar",
        type=int,
        default=64000,
        help="The graph compiler collar"
    )

    parser.add_argument(
        "--max-num-spks",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-overlap",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--total-max-overlaps",
        type=int,
        default=3600,
    )

    parser.add_argument(
        "--use-hat",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--use-layer-norm",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--use-large",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--downsample",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=25000000,
    )

    parser.add_argument(
        "--original-topo",
        type=str2bool,
        default=False
    )

    parser.add_argument(
        "--sort-strategy",
        type=str,
        default="start_time",
    )

    parser.add_argument(
        "--speaker-weight",
        type=float,
        default=1.0,
    )
    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - use_feat_batchnorm: Normalization for the input features, can be a
                              boolean indicating whether to do batch
                              normalization, or a float which means just scaling
                              the input features with this float value.
                              If given a float value, we will remove batchnorm
                              layer in `ConvolutionModule` as well.

        - attention_dim: Hidden dim for multi-head attention model.

        - head: Number of heads of multi-head attention model.

        - beam_size: It is used in k2.ctc_loss

        - reduction: It is used in k2.ctc_loss

        - use_double_scores: It is used in k2.ctc_loss

        - weight_decay:  The weight_decay for the optimizer.

        - warm_step: The warm_step for Noam optimizer.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            #"log_interval": 100, #100,
            "decode_hyp_interval": 100, #100,
            "reset_interval": 200,
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for zipformer
            "subsampling_factor": 4,  # not passed in, this is fixed.
            "warm_step": 2000,
            "env_info": get_env_info(),
            # parameters for loss
            #"beam_size": 24,
            "reduction": "sum",
            "use_double_scores": True,
            # parameters for decoding
            "search_beam": 20,
            "output_beam": 24,
            "min_active_states": 30,
            "max_active_states": 10000,
            "max_avg_arcs": 100000,
        }
    )

    return params


def get_encoder_model(params: AttributeDict) -> nn.Module:
    if params.use_large:
        bundle = torchaudio.pipelines.WAVLM_LARGE
    else:
        bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
    model = bundle.get_model()
    x = torch.rand(1, 400)
    odim = model(x)[0].size(-1)
    return model, odim


def get_mdctc_model(
    params: AttributeDict,
) -> nn.Module:
    encoder, encoder_dim = get_encoder_model(params)

    model = MDCTCModel(
        encoder=encoder,
        encoder_dim=encoder_dim,
        vocab_size1=params.vocab_size,
        vocab_size2=params.max_num_spks,
        hat=params.use_hat,
        layer_norm=params.use_layer_norm,
        downsample=params.downsample,
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

        if "cur_batch_idx" in saved_params:
            params["cur_batch_idx"] = saved_params["cur_batch_idx"]

    return saved_params


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

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_avg_speaker_density_per_example(start_frames_list, num_frames_list):
    """
    Args:
        start_frames_list: List of lists of segment start frames per example
        num_frames_list: List of lists of segment durations per example

    Returns:
        List of average speaker densities, one per example
    """
    avg_densities = []

    for start_frames, num_frames in zip(start_frames_list, num_frames_list):
        assert len(start_frames) == len(num_frames)
        if len(start_frames) == 0:
            avg_densities.append(0.0)
            continue

        end_frame = max(s + n for s, n in zip(start_frames, num_frames))
        density = np.zeros(end_frame, dtype=np.int32)

        for s, n in zip(start_frames, num_frames):
            density[s:s+n] += 1

        avg_density = density.mean()
        avg_densities.append(avg_density)

    return avg_densities


def align_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    graph_compiler: MDCTCGraphCompiler,
) -> Dict[str, List[List[str]]]:
    """
    Compute CTC loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Conformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      graph_compiler:
        It is used to build a decoding graph from a ctc topo and training
        transcript. The training transcript is contained in the given `batch`,
        while the ctc topo is built when this compiler is instantiated.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]
    max_density = batch["max_density"]
    feature_lens = batch["num_frames"].to(device)
    # at entry, feature is (N, T, C)
    feature = feature.to(device)
    feature_lens = feature_lens.to(device)
    texts = batch["texts"]
    speakers = batch["speakers"]
    sort_orders = batch["sort_orders"] 
    seq_idx = batch['supervisions']['sequence_idx']
    start_frames = [
        [batch['supervisions']['start_sample'][seq_idx == i][j].item() for j in sort_orders[i]]
        for i in range(seq_idx.max()+1)
    ]
    num_frames_init = [
        [batch['supervisions']['num_samples'][seq_idx == i][j].item() for j in sort_orders[i]]
        for i in range(seq_idx.max()+1)
    ]
    beam_factor = 1
    with torch.set_grad_enabled(False):
        ctc_output, x_lens = model(
            feature, feature_lens,
            speaker_weight = params.speaker_weight
        )
        subsampling_factor = params.subsampling_factor
        beam_size = params.beam_size
        reduction = params.reduction
        use_double_scores = params.use_double_scores
        
        # Align with WFST supervisions
        sequence_idx = torch.arange(
            0, x_lens.size(0),
        ).unsqueeze(0).t().to(torch.int32)

        start_frame = torch.zeros(
            [x_lens.size(0)], dtype=torch.int32,
        ).unsqueeze(0).t()

        num_frames = x_lens.unsqueeze(1).to(torch.int32).cpu()

        supervision_segments = torch.cat(
            [sequence_idx, start_frame, num_frames],
            dim=1,
        )
        supervision_segments = supervision_segments.to(torch.int32)

        collar = params.collar
        decoding_graphs = graph_compiler.compile(
            texts, start_frames, num_frames_init, speakers,
            collar=collar,
            max_overlaps=params.total_max_overlaps,
            original_topo=params.original_topo,
        )
        
        decoding_graphs = decoding_graphs.to(device)
        
        end_sup = time.time()

        dense_fsa_vec = k2.DenseFsaVec(
            ctc_output.float(),
            supervision_segments.cpu(),
            allow_truncate=subsampling_factor - 1,
        )

        lattice = k2.intersect_dense(
            a_fsas=decoding_graphs,
            b_fsas=dense_fsa_vec,
            output_beam=beam_size,
            max_states=params.max_states, #25000000,
            frame_idx_name='frame',
        )
      
        best_path = k2.shortest_path(lattice, use_double_scores=True)
        labels_t = best_path.labels.contiguous()
        frames_t = best_path.frame.contiguous()
        
        # arc_shape: collapses arc axis so shape corresponds to arc-level values per FSA
        arc_shape = best_path.arcs.shape().remove_axis(1)  # typically [B][num_arcs_in_path]

        # Build a RaggedTensor for all labels (no filtering yet)
        ragged_all_labels = k2.RaggedTensor(arc_shape, labels_t)

        # Remove all non-positive labels (<=0: blanks and possibly -1 sentinels)
        units = ragged_all_labels.remove_values_leq(0)
        # ragged_clean_labels.shape is now the correct ragged shape for the filtered values

        # Build the frames ragged using the SAME shape and the filtered frame values.
        # Compute boolean mask on the original flat tensors to get the same set of kept indices:
        keep_mask = (labels_t > 0)  # True for values we kept; this matches remove_values_leq(0)

        frames_kept = frames_t[keep_mask]

        # Now build ragged frames with the *new* shape
        ragged_frames = k2.RaggedTensor(units.shape, frames_kept)
        times = k2.RaggedTensor(ragged_frames.shape, ragged_frames.values * params.frame_duration)
        spks = k2.RaggedTensor(units.shape, units.values // (params.vocab_size - 1))
        syms = k2.RaggedTensor(units.shape, units.values.remainder((params.vocab_size - 1)))
 
        cut_ids = [c.id for c in batch["supervisions"]["cut"]]
        for i in range(times.shape.dim0):
            spk2int = {k: s_i for s_i, k in enumerate(dict.fromkeys(speakers[i]))}
            int2spk = {s_i: k for k, s_i in spk2int.items()}

            t, s, u = times[i], spks[i], syms[i]
            if t.numel() == 0:
                yield cut_ids[i], None
                continue
            
            max_dur = (feature_lens[i] / 16000 - t[-1]).item()
            avg_token_durs = {}
            for j in range(params.max_num_spks):
                if t[s == j].numel() <= 1:
                    avg_token_durs[j] = 0.2
                else:
                    avg_token_durs[j] = round(
                        torch.mean(
                            (torch.roll(t[s == j], shifts=-1) - t[s == j])[:-1]
                        ).item(), 3
                    )
            for j in avg_token_durs:
                if math.isnan(avg_token_durs[j]):
                    avg_token_durs[j] = 0.2
            durs = (torch.roll(t, shifts=-1) - t)[:-1].tolist()
            durs = [round(val, 3) for val in durs]  
            durs.append(min(avg_token_durs[s[-1].item()], max_dur))
            for d in durs:
                assert d > 0
            pieces = [
                graph_compiler.sp.id_to_piece(p)
                for p in u.tolist()
            ] 
            result = list(
                zip(
                    [int2spk[s_idx] for s_idx in s.tolist()],
                    pieces,
                    [round(t_, 3) for t_ in t.tolist()],
                    durs,
                )
            )
            #result = list(
            #    zip(
            #        [speakers[i][s_idx] for s_idx in s.tolist()],
            #        pieces,
            #        [round(t_, 3) for t_ in t.tolist()],
            #        durs,
            #    )
            #)
            yield cut_ids[i], result 


def align_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    graph_compiler: MDCTCGraphCompiler,
):
    """Align dataset
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    alignments = {}
    failed = 0
    total = 0
    for batch_idx, batch in tqdm(enumerate(dl)):
        texts = batch["supervisions"]["text"]
        for c_id, ali in align_one_batch(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                batch=batch,
            ):
            total += 1
            if ali is None:
                logging.info(f"Alignment {c_id} failed. Skipping ... ")
                failed += 1
                continue
            c_id_nospk = c_id.rsplit("-", 1)[0]
            if c_id_nospk not in alignments:
                alignments[c_id_nospk] = ali
            else:
                alignments[c_id_nospk] += ali
        batch_str = f"{batch_idx}/{num_batches}"
        logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    for k in alignments:
        alignments[k] = sorted(alignments[k], key=lambda x: x[2])
    logging.info(f"Done. Success for {(total-failed)} / {total}.") 
    alignments = sorted(alignments.items(), key=lambda x: x[0])
    return alignments

 
@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log-argmax-decode/log-decode")
    logging.info("Decoding started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    graph_compiler = MDCTCGraphCompiler(
        params.lang_dir,
        device='cpu',
    )
    
    params.vocab_size = graph_compiler.sp.vocab_size()
    
    logging.info("About to create model")

    model = get_mdctc_model(params)
    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    model.eval()
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    librispeech = LibriSpeechAsrDataModule(args)

    synth_cuts = librispeech.synth2_cuts(2)

    synth_dl = librispeech.mix_to_mono_dataloader(synth_cuts)
    
    test_sets = ["synth2",]
    test_dl = [synth_dl,]

    for set, dl in zip(test_sets, test_dl):
        alignments = align_dataset(
            dl=dl,
            params=params,
            model=model,
            graph_compiler=graph_compiler,
        )
    
        ali_dir = params.exp_dir / "align"
        ali_dir.mkdir(parents=True, exist_ok=True)
        fname = f"alignments_mono_chkpt{params.iter}_avg{params.avg}_{set}_collar{params.collar}_{params.suffix}.json"
        with open(ali_dir / fname, "w") as f:
            json.dump(alignments, f, indent=4)			

    logging.info("Done!")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


if __name__ == "__main__":
    main()
