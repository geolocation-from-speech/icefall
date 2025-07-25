#!/usr/bin/env python3
# Copyright 2021 Xiaomi Corporation (Author: Liyong Guo, Fangjun Kuang)
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


import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
from mdctc_graph_compiler2 import MDCTCGraphCompiler
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from asr_datamodule import LibriSpeechAsrDataModule
from train import (
    get_mdctc_model,
    get_params,
    compute_avg_speaker_density_per_example,
)

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import (
    AttributeDict,
    load_averaged_model,
    setup_logger,
    str2bool,
)

import asclite
import time
from tqdm import tqdm
import json


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
        default="zipformer_ctc/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_bpe_500",
        help="The lang dir",
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
        "--collar",
        type=int,
        default=64000,
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
    
    parser.add_argument(
        "--frame-duration",
        type=int,
        default=0.04,
    )

    return parser


def get_nonzero_span_starts(batch, params):
    # batch shape: [B, T]
    B, T = batch.shape

    # Shift the input right by 1 along time dimension, prepend zeros
    shifted = torch.cat(
        [
            torch.zeros(B, 1, dtype=batch.dtype).to(batch.device),
            batch[:, :-1]
        ], dim=1
    )

    # Find where a new span starts: nonzero & (value != previous or previous was zero)
    is_new_span = (batch != 0) & ((shifted != batch) | (shifted == 0))

    # For each batch, get the indices where this is true
    result = [
        torch.nonzero(seq, as_tuple=False).squeeze(1) * params.frame_duration
        for seq in is_new_span
    ]

    return result


def merge_bpe_with_times(sp, tokens, times, max_time):
    assert len(tokens) == len(times), "Tokens and times must match in length"
    words = []
    current_word = ''
    current_time = None

    for token_, time in zip(tokens, times):
        token = sp.id_to_piece(token_.item())
        if token.startswith('▁') or not current_word:  # new word
            if current_word:
                words.append(
                    (
                        current_word,
                        round(current_time, 2),
                        round(time - current_time, 2)
                    )
                )
            current_word = token.lstrip('▁')
            current_time = time
        else:
            current_word += token
    if current_word:
        words.append(
            (
                current_word,
                round(current_time, 2),
                round(min(time + 0.5, max_time.item()) - current_time, 2)
            )
        )

    return words


def hyp_to_stm(hyp, cut_idx):
    utts = []
    for s in hyp:
        entry = {
            "session_id": cut_idx,
            "words": " ".join([w[0] for w in hyp[s]]),
            "speaker": str(s),
            "start_time": round(hyp[s][0][1], 2),
            "end_time": round(hyp[s][-1][1] + hyp[s][-1][2], 2),
        }
        utts.append(entry)
    return utts


def ref_to_stm(ref, cut_idx):
    utts = []
    for s in ref:
        entry = {
            "session_id": cut_idx,
            "words": ref[s][0],
            "speaker": str(s),
            "start_time": round(ref[s][1], 2),
            "end_time": round(ref[s][1] + ref[s][2], 2),
        }
        utts.append(entry)
    return utts


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    graph_compiler: MDCTCGraphCompiler,
    batch: dict,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if no rescoring is used, the key is the string `no_rescore`.
               If LM rescoring is used, the key is the string `lm_scale_xxx`,
               where `xxx` is the value of `lm_scale`. An example key is
               `lm_scale_0.7`
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.

        - params.method is "1best", it uses 1best decoding without LM rescoring.
        - params.method is "nbest", it uses nbest decoding without LM rescoring.
        - params.method is "nbest-rescoring", it uses nbest LM rescoring.
        - params.method is "whole-lattice-rescoring", it uses whole lattice LM
          rescoring.

      model:
        The neural model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict. Note: If it decodes to nothing, then return None.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]
    feature_lens = batch["num_frames"].to(device)
    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature = feature.to(device)
    texts = [[t.strip() for t in b] for b in batch["texts"]]
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

    with torch.set_grad_enabled(False):
        start = time.time()
        ctc_output, x_lens = model(
            feature,
            feature_lens,
        )
        end_nnet = time.time()
        subsampling_factor = params.subsampling_factor
        use_double_scores = params.use_double_scores


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
    #decoding_graphs = graph_compiler.compile_transcript(texts, start_frames, num_frames_init)

    ctc_output[..., 0] -= params.blank_weight
    preds = ctc_output.argmax(-1)
    times = get_nonzero_span_starts(preds, params)
    hyps = [
        preds[i].unique_consecutive(dim=-1)[preds[i].unique_consecutive(dim=-1) != 0].squeeze(0)
        for i in range(preds.size(0))
    ]
    spks = [
        hyps[i] // (params.vocab_size - 1)
        for i in range(preds.size(0))
    ]
    units = [
        hyps[i].remainder((params.vocab_size - 1))
        for i in range(preds.size(0))
    ]
    
    cut_ids = [c.id for c in batch["supervisions"]["cut"]]
    tokens = {}
    for i, (cut_id, s, u, t) in enumerate(zip(cut_ids, spks, units, times)):
        spk2int = {k: j for j, k in enumerate(dict.fromkeys(speakers[i]))}
        int2spk = {j: k for k, j in spk2int.items()}
        speaker_hyps = {}
        speaker_refs = {}
        for s_i in range(params.max_num_spks): 
            if s_i in s:
                speaker_hyps[s_i] = merge_bpe_with_times(
                    graph_compiler.sp,
                    u[s == s_i],
                    [t_.item() for t_ in t[s == s_i]],
                    x_lens[i]*params.frame_duration
                )
            if s_i in int2spk: 
                text_i = []
                found_start = False
                curr_dur = 0
                for j, (text, spk) in enumerate(zip(texts[i], speakers[i])):
                    if spk2int[spk] == s_i:
                        if not found_start:
                            start_i = start_frames[i][j]
                            found_start = True
                        curr_dur = start_frames[i][j] + num_frames_init[i][j]
                        text_i.append(text)
                text_i = " ".join(text_i)
                speaker_refs[s_i] = (
                    graph_compiler.sp.decode(text_i),
                    start_i * (1/16000),
                    curr_dur * (1/16000),
                )
        stm_hyp = hyp_to_stm(speaker_hyps, cut_id)
        stm_ref = ref_to_stm(speaker_refs, cut_id)
        yield stm_hyp, stm_ref


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    graph_compiler: MDCTCGraphCompiler,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      graph_compiler:
        The graph compiler to make the shuffle automata
    Returns:
      Return a dict, whose key may be "no-rescore" if no LM rescoring
      is used, or it may be "lm_scale_0.7" if LM rescoring is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = defaultdict(list)
    hyps, refs = [], []
    for batch_idx, batch in tqdm(enumerate(dl)):
        texts = batch["supervisions"]["text"]
        for hyps_, refs_ in decode_one_batch(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                batch=batch,
            ):
            hyps.extend(hyps_)
            refs.extend(refs_)
        batch_str = f"{batch_idx}/{num_batches}"
        logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    hyps = sorted(hyps, key=lambda x: (x["session_id"], x["start_time"]))
    refs = sorted(refs, key=lambda x: (x["session_id"], x["start_time"]))
    return hyps, refs



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
        collar=params.collar,
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

    #valid_cuts = librispeech.synth_cuts()
    #test_other_cuts, test_clean_cuts = librispeech.single_speaker_cuts()
    #valid_cuts = librispeech.libricss_cuts()
    valid_cuts = librispeech.libri2mix_test_both_cuts()
    valid_dl = librispeech.valid_dataloaders(valid_cuts)

    test_sets = ["valid",]
    test_dl = [valid_dl,]

    for set, dl in zip(test_sets, test_dl):
        hyps, refs = decode_dataset(
            dl=dl,
            params=params,
            model=model,
            graph_compiler=graph_compiler,
        )
    
    decode_dir = params.exp_dir / "decode"
    decode_dir.mkdir(parents=True, exist_ok=True)
    with open(decode_dir / f"hyps_chkpt{params.iter}_avg{params.avg}_{params.suffix}.stm", "w") as f:
        for l in hyps:
            print(f"{l['session_id']} 1 {l['speaker']} {l['start_time']} {l['end_time']} {l['words']}", file=f)
    
    with open(decode_dir / f"refs_chkpt{params.iter}_avg{params.avg}_{params.suffix}.stm", "w") as f:
        for l in refs:
            print(f"{l['session_id']} 1 {l['speaker']} {l['start_time']} {l['end_time']} {l['words']}", file=f)

    logging.info("Done!")


if __name__ == "__main__":
    main()
