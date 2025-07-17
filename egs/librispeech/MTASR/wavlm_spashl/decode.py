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
        default=0,
    )

    parser.add_argument(
        "--collar",
        type=int,
        default=64000,
    )

    return parser


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
    texts = batch["texts"]
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
    decoding_graphs = graph_compiler.compile_transcript(texts, start_frames, num_frames_init)

    ctc_output[..., 0] -= params.blank_weight
    preds = ctc_output.argmax(-1)
    hyps = [
        preds[i].unique_consecutive()[preds[i].unique_consecutive() != 0].squeeze(0).tolist()
        for i in range(preds.size(0))
    ]

    cut_ids = [c.id for c in batch["supervisions"]["cut"]]
    errors, total, ter, alis, dels, ins, subs, corr = asclite.compute_ter(decoding_graphs, hyps)
    tokens = {}
    for idx, (hyp_ids, ref_ids) in enumerate(alis):
        tokens_ = []
        for i, j in zip(hyp_ids, ref_ids):
            h = graph_compiler.sp.id_to_piece(i)
            r = graph_compiler.sp.id_to_piece(j)
            tokens_.append([h, r])
        tokens[cut_ids[idx]] = tokens_
    return errors, total, ter, tokens, dels, ins, subs, corr


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
    ters = []
    errors = 0
    total = 0
    dels, ins, subs, corr = 0, 0, 0, 0
    tokens = {}
    for batch_idx, batch in tqdm(enumerate(dl)):
        texts = batch["supervisions"]["text"]
        errors_, total_, ter, tokens_, d, i, s, c = decode_one_batch(
            params=params,
            model=model,
            graph_compiler=graph_compiler,
            batch=batch,
        )
        ters.append(ter)
        tokens.update(tokens_)
        errors += errors_
        total += total_
        dels += d
        ins += i
        subs += s
        corr += c
        batch_str = f"{batch_idx}/{num_batches}"
        logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
        logging.info(f"ter: {ter}")
    results["ter"] = errors / total
    results["ter_indv"] = ters
    results["errors"] = errors
    results["total"] = total
    results["dels"] = dels
    results["ins"] = ins
    results["subs"] = subs
    results["corr"] = corr
    return results, tokens



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

    valid_cuts = librispeech.synth_cuts()
    #valid_cuts = librispeech.libricss_cuts()
    valid_dl = librispeech.valid_dataloaders(valid_cuts)

    test_sets = ["valid",]
    test_dl = [valid_dl,]

    for set, dl in zip(test_sets, test_dl):
        results_dict, tokens = decode_dataset(
            dl=dl,
            params=params,
            model=model,
            graph_compiler=graph_compiler,
        )
    
    decode_dir = params.exp_dir / "decode"
    decode_dir.mkdir(parents=True, exist_ok=True)
    with open(decode_dir / f"results_chkpt{params.iter}_avg{params.avg}_{params.suffix}_collar{params.collar}.json", "w") as f:
        json.dump(results_dict, f, indent=4)    
    
    with open(decode_dir / f"alignments_chkpt{params.iter}_avg{params.avg}_{params.suffix}.txt", "w") as f:
        for uttid in tokens:
            seq = tokens[uttid]
            hyp_tokens = [pair[0] for pair in seq]
            ref_tokens = [pair[1] for pair in seq]
            widths = [max(len(h), len(r)) for h, r in zip(hyp_tokens, ref_tokens)]
            # Pad each token to column width
            ref_line = "ref: " + "  ".join(r.ljust(w) for r, w in zip(ref_tokens, widths))
            hyp_line = "hyp: " + "  ".join(h.ljust(w) for h, w in zip(hyp_tokens, widths))
            print(ref_line, file=f)
            print(hyp_line, file=f)
            print("", file=f)
	
    with open(decode_dir / f"alignments_chkpt{params.iter}_avg{params.avg}_{params.suffix}.json", "w") as f:
        json.dump(tokens, f, indent=4)
     
    logging.info("Done!")


if __name__ == "__main__":
    main()
