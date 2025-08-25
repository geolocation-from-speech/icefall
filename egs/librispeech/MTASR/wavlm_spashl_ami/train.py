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
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import os
import time
import torch
import k2
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from decoder import Decoder
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from lhotse import CutSet
from model import MDCTCModel
from optim import Eden, LRScheduler, ScaledAdam
from torch.optim import Adam
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from zipformer import Zipformer

from icefall import diagnostics
#from mdctc_graph_compiler import MDCTCGraphCompiler
from mdctc_graph_compiler2 import MDCTCGraphCompiler
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
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


LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, LRScheduler]


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
        "--base-lr", type=float, default=0.05, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=5000,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=3.5,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
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
        end of each epoch where `xxx` is the epoch number counting from 0.
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
        "--collar",
        type=int,
        default=250,
        help="The graph compiler collar"
    )

    parser.add_argument(
        "--freeze-lr", type=float, default=0.0001,
        help="The learning rate used to train new unfrozen parameters that are "
        "introduced on top of the base wav2vec2 model. We normally first train "
        "these parameters for some number of iterations/"
    )
    
    parser.add_argument("--freeze-iters", type=int, default=500,
        help="The number of iterations the base wav2vec2 model is frozen."
    )

    parser.add_argument(
        "--pct-start", type=float, default=0.08, help="The percent of steps "
        "to use as warmup",
    )

    parser.add_argument(
        "--total-steps", type=int, default=100000, help="The total number of "
        "steps for the lr_scheduler",
    )    
    
    parser.add_argument(
        "--weight-decay", type=float, default=1e-08, help="Adam weight decay",
    )

    parser.add_argument(
        "--max-num-spks",
        type=int,
        default=2,
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

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - beam_size: It is used in k2.ctc_loss

        - reduction: It is used in k2.ctc_loss

        - use_double_scores: It is used in k2.ctc_loss

        - weight_decay:  The weight_decay for the optimizer.

        - warm_step: The warm_step for Noam optimizer.
    """
    params = AttributeDict(
        {
            "frame_shift_ms": 10.0,
            "allowed_excess_duration_ratio": 0.1,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 1,
            "decode_hyp_interval": 20,
            "reset_interval": 200,
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for zipformer
            "feature_dim": 80,
            "subsampling_factor": 4,  # not passed in, this is fixed.
            "warm_step": 2000,
            "env_info": get_env_info(),
            # parameters for loss
            "beam_size": 24,
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


def compute_loss(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    graph_compiler: MDCTCGraphCompiler,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
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
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
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
    seq_idx = batch['supervisions']['sequence_idx']
    start_frames = [
        batch['supervisions']['start_sample'][seq_idx == i].tolist()
        for i in range(seq_idx.max()+1)
    ]
    num_frames_init = [
        batch['supervisions']['num_samples'][seq_idx == i].tolist()
        for i in range(seq_idx.max()+1)
    ]
    #beam_factor = max(0.3, (100000 - params.batch_idx_train)/100000)
    beam_factor = 1
    with torch.set_grad_enabled(is_training):
        start = time.time()
        ctc_output, x_lens = model(feature, feature_lens)
        ctc_output -= 1e-09
        end_nnet = time.time()
        subsampling_factor = params.subsampling_factor
        beam_size = params.beam_size * beam_factor
        reduction = params.reduction
        use_double_scores = params.use_double_scores


        if params.batch_idx_train % params.decode_hyp_interval == 0:
            preds = ctc_output.argmax(-1)
            hyps = []
            spk_hyps = []
            for i in range(preds.size(0)):
                hyps_ = {}
                preds_no_repeats = preds[i].unique_consecutive()
                preds_no_repeats_or_blanks = preds_no_repeats[preds_no_repeats != 0].squeeze()
                spks = preds_no_repeats_or_blanks // (params.vocab_size - 1)
                units = preds_no_repeats_or_blanks.remainder((params.vocab_size - 1))
                for s in range(params.max_num_spks):
                    s_units = units[spks == s].tolist()
                    hyps_[s] = s_units
                hyps.append(hyps_)
                spk_hyps.append(spks.view(-1).tolist())

            # Construct the speaker ref
            text_lens = []
            num_frames_ref, num_tokens_ref = 0, 0
            for t_idx in range(len(texts[0])): 
                text_lens.append(start_frames[0][t_idx] + num_frames_init[0][t_idx])
                num_frames_ref += num_frames_init[0][t_idx]
                num_tokens_ref += len(graph_compiler.sp.encode(texts[0][t_idx]))
            max_len = max(text_lens)
            frames_per_token = num_frames_ref // num_tokens_ref + 1
            start_tokens = [start_frames[0][t_idx] // frames_per_token for t_idx in range(len(texts[0]))]
            spk2int = {k: i for i, k in enumerate(dict.fromkeys(speakers[0]))}

            logging.info(f" ===================================== " )
            logging.info(f"spk_ref: ")
            for t_idx, st in enumerate(start_tokens):
                blanks = " "*st
                tokens = str(spk2int[speakers[0][t_idx]])*len(graph_compiler.sp.encode(texts[0][t_idx]))
                logging.info(f"{blanks}{tokens}")
            logging.info(f" ---------------")
            logging.info(f"spk_hyp: {"".join(map(str, spk_hyps[0]))}")
            for t_idx in range(len(hyps[0])):
                logging.info(f"hyp_{t_idx}: {graph_compiler.sp.decode(hyps[0][t_idx])}")
                logging.info(f"")
            logging.info(f" --------")
            for t_idx, text in enumerate(texts[0]):
                logging.info(f"ref_{t_idx}: {text}")
                logging.info(f"")
            logging.info("--------------------------------------")
        
        # Decoding with WFST supervisions
        sequence_idx = torch.arange(
            0, x_lens.size(0),
        ).unsqueeze(0).t().to(torch.int32)

        start_frame = torch.zeros(
            [x_lens.size(0)], dtype=torch.int32,
        ).unsqueeze(0).t()

        num_frames = x_lens.unsqueeze(1).to(torch.int32).cpu()
        #num_frames = (x_lens * 2).unsqueeze(1).to(torch.int32).cpu()
        #num_frames = x_lens.unsqueeze(1).to(torch.int32).cpu()

        supervision_segments = torch.cat(
            [sequence_idx, start_frame, num_frames],
            dim=1,
        )
        supervision_segments = supervision_segments.to(torch.int32)

        # Works with a BPE model
        densities = compute_avg_speaker_density_per_example(start_frames, num_frames_init)
        
        collar = params.collar
        decoding_graphs = graph_compiler.compile(
            texts, start_frames, num_frames_init, speakers,
            collar=collar, dynamic_collar=True, max_overlaps=4000,
        )
        num_arcs = decoding_graphs.labels.size(0) / len(texts)
        logging.info(f"Num arcs: {num_arcs}")
        logging.info(f"Max Spk density: {max_density}")
        logging.info(f"Length: {feature_lens[0].item()}")
        #while num_arcs > params.max_avg_arcs:
        #    collar = collar // 2 
        #    logging.warning(f"Too many arcs. Halving collar: {collar}")
        #    decoding_graphs = graph_compiler.compile(
        #        texts, start_frames, num_frames_init, speakers,
        #        collar=collar
        #    )
        #    num_arcs = decoding_graphs.labels.size(0)
        #    logging.info(f"Num arcs: {num_arcs}")

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
            max_states=25000000,
            frame_idx_name=None,
        )
       
        #import pdb; pdb.set_trace() 
        #empty_idxs = []
        #for i in range(lattice.shape[0]):
        #    if lattice[i].labels.size(0) == 0:
        #        empty_idxs.append(i)
        #lattice = k2.intersect_dense_pruned(
        #    decoding_graphs,
        #    dense_fsa_vec,
        #    search_beam=15.0,
        #    output_beam=beam_size,
        #    min_active_states=30,
        #    max_active_states=50000, 
        #)
       
        tot_scores = lattice.get_tot_scores(
            log_semiring=True,
            use_double_scores=use_double_scores
        )
        loss = -1 * tot_scores
        loss = loss[~torch.isinf(loss)]
        if torch.any(loss < 0):
            logging.info("Negative loss. Clamping") 
            loss = torch.clamp(loss, min=0.0)
        loss = loss.to(torch.float32) 
        ctc_loss = loss.sum() 
        #ctc_loss = k2.ctc_loss(
        #    decoding_graph=decoding_graphs,
        #    dense_fsa_vec=dense_fsa_vec,
        #    output_beam=beam_size,
        #    reduction=reduction,
        #    use_double_scores=use_double_scores,
        #)
        end_ctc = time.time()
        
    nnet_time = end_nnet - start
    sup_time = end_sup - end_nnet
    ctc_time = end_ctc - end_sup
    total_time = end_ctc - start
    tot_frames = num_frames.sum().item()
    info = MetricsTracker()
    info["frames"] = tot_frames
    info["ctc_loss"] = ctc_loss.detach().cpu().item()
    info["beam_size"] = beam_size * tot_frames 
    info["max_length"] = num_frames.max() * tot_frames
    info["max_duration"] = num_frames.max() * 0.01 * params.subsampling_factor * tot_frames
    avg_num_texts = sum([len(t) for t in texts])/len(texts)
    info["num_texts"] = avg_num_texts * tot_frames
    info["spk_density"] = (sum(densities) / len(densities)) * tot_frames
    info["arc_density"] = decoding_graphs.arcs.values().size(0)
    info["pct_ctc"] = ctc_time / total_time * tot_frames
    info["pct_nnet"] = nnet_time / total_time * tot_frames
    info["pct_sup"] = sup_time / total_time * tot_frames
    loss = ctc_loss
    assert loss.requires_grad == is_training, f"{loss.requires_grad} != {is_training}"
    info["loss"] = loss.detach().cpu().item()

    # `utt_duration` and `utt_pad_proportion` would be normalized by `utterances`  # noqa
    info["utterances"] = feature.size(0)
    # averaged input duration in frames over utterances
    info["utt_duration"] = feature_lens.sum().item()
    # averaged padding proportion over utterances
    info["utt_pad_proportion"] = (
        ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
    )
    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: nn.Module,
    graph_compiler: MDCTCGraphCompiler,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            batch=batch,
            graph_compiler=graph_compiler,
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
    graph_compiler: MDCTCGraphCompiler,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    init_lr: float, 
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      graph_compiler:
        It is used to convert transcripts to FSAs.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
    """
    model.train()

    tot_loss = MetricsTracker()
    
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    autocast_args = {'device_type': 'cuda'} if device.type == 'cuda' else {'device_type': 'cpu'}

    start = time.time()
    num_egs = 0
    num_frames = 0 
    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = batch["num_frames"].size(0)
        is_frozen = model.module.frozen if isinstance(model, DDP) else model.frozen
        unfreeze = model.module.unfreeze_encoder if isinstance(model, DDP) else model.unfreeze_encoder
        if params.batch_idx_train > params.freeze_iters and is_frozen:
            logging.info("Unfreezing ...")
            unfreeze()
            for param_group in optimizer.param_groups:
                param_group['lr'] = init_lr

        try:
            #with torch.cuda.amp.autocast(enabled=params.use_fp16):
            with torch.amp.autocast(**autocast_args):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    graph_compiler=graph_compiler,
                    is_training=True,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            #if loss.isinf():
            #    logging.warning(f"Infinite Loss {params.batch_idx_train}")
            #    continue;
            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()
            if not is_frozen:
                scheduler.step()
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except:  # noqa
            #display_and_save_batch(batch, params=params, graph_compiler=graph_compiler)
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
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
            params.cur_batch_idx = batch_idx
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
            del params.cur_batch_idx
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
            if cur_grad_scale < 1.0 or (cur_grad_scale < 8.0 and batch_idx % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                raise_grad_scale_is_too_small_error(cur_grad_scale)

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
                        "train/grad_scale",
                        cur_grad_scale,
                        params.batch_idx_train,
                    )

        if batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
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
    #local_rank = int(os.environ.get("LOCAL_RANK", 0))
    #torch.cuda.set_device(local_rank)
    params = get_params()
    params.update(vars(args))
    #params.master_port = int(os.environ.get("MASTER_PORT", 12355))
    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")
    logging.info(params)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None


    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    graph_compiler = MDCTCGraphCompiler(
        params.lang_dir,
        device='cpu',
        #collar=params.collar,
    )

    params.vocab_size = graph_compiler.sp.vocab_size()
    
    logging.info("About to create model")

    model = get_mdctc_model(params)
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)
 
    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
     
    parameters_names = []
    parameters_names.append(
        [name_param_pair[0] for name_param_pair in model.named_parameters()]
    )
    
    optimizer = Adam(
        list(filter(lambda p: p.requires_grad, model.parameters())),
        lr=params.base_lr,
        betas=(0.9, 0.98), eps=1e-08, weight_decay=params.weight_decay, 
    )
 
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=params.base_lr,
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

    #optimizer = ScaledAdam(
    #    model.parameters(),
    #    lr=params.base_lr,
    #    clipping_scale=2.0,
    #    parameters_names=parameters_names,
    #)

    #scheduler = Eden(
    #    optimizer,
    #    params.lr_batches,
    #    params.lr_epochs,
    #    warmup_batches=1000
    #)

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
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    args.return_cuts = False
    librispeech = LibriSpeechAsrDataModule(args)
    train_cuts = librispeech.train_cuts()
    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None


    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        return (
            1.0 <= c.duration <= 62.0 and len(CutSet([c]).speakers) < params.max_num_spks
            #and sum(len(s.text) for s in c.supervisions) <= 400
        )

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    train_dl = librispeech.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )
    
    valid_cuts = librispeech.valid_cuts()
    valid_cuts = valid_cuts.filter(remove_short_and_long_utt)
    valid_dl = librispeech.valid_dataloaders(valid_cuts)

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    fix_random_seed(params.seed)
    train_dl.sampler.set_epoch(0)

    params.cur_epoch = 0

    train_one_epoch(
        params=params,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
        graph_compiler=graph_compiler,
        train_dl=train_dl,
        valid_dl=valid_dl,
        scaler=scaler,
        tb_writer=tb_writer,
        world_size=world_size,
        rank=rank,
        init_lr=init_lr,
    )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()

 
#def display_and_save_batch(
#    batch: dict,
#    params: AttributeDict,
#    graph_compiler: MDCTCGraphCompiler,
#) -> None:
#    """Display the batch statistics and save the batch into disk.
#
#    Args:
#      batch:
#        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
#        for the content in it.
#      params:
#        Parameters for training. See :func:`get_params`.
#      graph_compiler:
#        It is used to build a decoding graph from a ctc topo and training
#        transcript. The training transcript is contained in the given `batch`,
#        while the ctc topo is built when this compiler is instantiated.
#    """
#    from lhotse.utils import uuid4
#
#    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
#    logging.info(f"Saving batch to {filename}")
#    torch.save(batch, filename)
#
#    supervisions = batch["supervisions"]
#    features = batch["inputs"]
#
#    logging.info(f"features shape: {features.shape}")
#
#    y = graph_compiler.texts_to_ids(supervisions["text"])
#    num_tokens = sum(len(i) for i in y)
#    logging.info(f"num tokens: {num_tokens}")


def scan_pessimistic_batches_for_oom(
    model: nn.Module,
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    graph_compiler: MDCTCGraphCompiler,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 0 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            optimizer.zero_grad()
            loss, _ = compute_loss(
                params=params,
                model=model,
                batch=batch,
                graph_compiler=graph_compiler,
                is_training=True,
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0, 2.0)
            optimizer.step()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            raise


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)
#    rank = int(os.environ.get("RANK", 0))
#    world_size = int(os.environ.get("WORLD_SIZE", 1))
#    args.master_port = int(os.environ.get("MASTER_PORT", 12355))
#    run(rank=rank, world_size=world_size, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


if __name__ == "__main__":
    main()
