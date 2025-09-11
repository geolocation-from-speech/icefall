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
from mdctc_graph_compiler import MDCTCGraphCompiler
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

from lhotse import CutSet
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
        default=1,
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

    parser.add_argument(
        "--masked-spks",
        type=str,
        default=None
    )

    parser.add_argument(
        "--collar",
        type=int,
        default=32000,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=24,
    )



    return parser



def align_one_batch(
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
    speaker_mask = None if params.masked_spks is None else [int(s) for s in params.masked_spks.split()]
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
            speaker_mask = speaker_mask,
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
    
    collar = params.collar
    # Works with a BPE model
    densities = compute_avg_speaker_density_per_example(start_frames, num_frames_init)
    decoding_graphs = graph_compiler.compile(
        texts, start_frames, num_frames_init, speakers,
        collar=collar, dynamic_collar=True, max_overlaps=5000
    )

    decoding_graphs = decoding_graphs.to(device)
    
    dense_fsa_vec = k2.DenseFsaVec(
            ctc_output.float(),
            supervision_segments.cpu(),
            allow_truncate=subsampling_factor - 1,
    )

    lattice = k2.intersect_dense(
        a_fsas=decoding_graphs,
        b_fsas=dense_fsa_vec,
        output_beam=params.beam_size,
        max_states=25000000,
        frame_idx_name=None,
    )
    best_path = k2.shortest_path(lattice, use_double_scores=True)
    cut_ids = [c.id for c in batch["supervisions"]["cut"]]
    alignments = {}
    for i in range(best_path.shape[0]):
        if best_path[i].labels.numel() == 0:
            logging.info(f"Empty alignment. Skipping {cut_ids[i]} ...")
            continue
        t = best_path[i].aux_labels[:-1] 
        nonzero_indices = torch.nonzero(t, as_tuple=False).squeeze().view(-1)
        nonzero_values = t[nonzero_indices]
        units = nonzero_values.remainder((params.vocab_size - 1)).view(-1)
        spks = (nonzero_values // (params.vocab_size - 1)).view(-1)

        # Compute steps to next non-zero
        next_nonzero = torch.roll(nonzero_indices, shifts=-1).view(-1)
        steps = next_nonzero - nonzero_indices.view(-1)
        steps[-1] = -1  # Last element has no next non-zero
        steps -= 1 # just to pad

        try:
            pieces = [
                graph_compiler.sp.id_to_piece(p)
                for p in units.tolist()
            ]
        except:
            import pdb; pdb.set_trace()

        starts = [
            round(s, 1)
            for s in (nonzero_indices * params.frame_duration).tolist()
        ]

        durs = [
            round(d + 0.2, 1)
            for d in (steps * params.frame_duration).tolist()
        ]
        
        # Combine into list of tuples
        result = list(zip(pieces, starts, durs))
        alignments[cut_ids[i]] = result
    
    return alignments 


def align_dataset(
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

    alignments = {}
    for batch_idx, batch in tqdm(enumerate(dl)):
        texts = batch["supervisions"]["text"]
        alis = align_one_batch(
            params=params,
            model=model,
            graph_compiler=graph_compiler,
            batch=batch,
        )
        alignments.update(alis)
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

    #valid_cuts = librispeech.synth_cuts()
    #valid_cuts = librispeech.libri2mix_test_both_cuts()
    #test_other_cuts, test_clean_cuts = librispeech.single_speaker_cuts()
    #valid_cuts = librispeech.libricss_cuts()
    ami_dev = librispeech.ami_dev()
    ami_dev = ami_dev.filter(lambda c: len(CutSet([c]).speakers) <= params.max_num_spks)
    valid_dl = librispeech.valid_dataloaders(ami_dev)

    test_sets = ["ami_dev",]
    test_dl = [valid_dl,]

    for set, dl in zip(test_sets, test_dl):
        alignments = align_dataset(
            dl=dl,
            params=params,
            model=model,
            graph_compiler=graph_compiler,
        )
    
        ali_dir = params.exp_dir / "align"
        ali_dir.mkdir(parents=True, exist_ok=True)
        fname = f"alignments_chkpt{params.iter}_avg{params.avg}_{set}_collar{params.collar}_{params.suffix}.json"
        with open(ali_dir / fname, "w") as f:
            json.dump(alignments, f, indent=4)			

    logging.info("Done!")


if __name__ == "__main__":
    main()
