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
from typing import Dict, List, Tuple

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

from icefall.lexicon import Lexicon

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

from icefall.decode import (
    get_lattice,
    one_best_decoding,
    rescore_with_whole_lattice,
    rescore_with_n_best_list,
    nbest_rescore_with_LM,
)

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
        "--acoustic-weight",
        type=float,
        default=3.0
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
        "--downsample",
        type=str2bool,
        default=True,
    )
    
    parser.add_argument(
        "--frame-duration",
        type=float,
        default=0.04,
    )

    parser.add_argument(
        "--target-spks",
        type=str,
        default=None
    )

    parser.add_argument(
        "--test-sets",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--sort-strategy",
        type=str,
        default="start_time",
    )

    parser.add_argument(
        "--modified",
        type=str2bool,
        default=None,
    )

    parser.add_argument(
        "--lm-dir",
        type=str,
        default="data/lm",
        help="N-gram model (.fst.txt) for rescoring"
    )

    parser.add_argument(
        "--rescore",
        type=str2bool,
        default=False,
        help="Whether or not to rescore"
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=100,
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
                        round(min(current_time, max_time.item()), 2),
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
                round(min(current_time, max_time.item()), 2),
                round(max(0, time - current_time), 2)
            )
        )

    return words


def hyp_to_stm(hyp, cut_idx):
    utts = []
    for s in hyp:
        if len(hyp[s]) == 0:
            continue
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
    graph,
    rescore,
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
    target_spks = [i for i in range(params.max_num_spks)]
    if params.target_spks is not None:
        target_spks = [int(s) for s in params.target_spks.split()]
    
    seq_idx = batch['supervisions']['sequence_idx']
    sort_orders = batch["sort_orders"] 
    start_frames = [
        [batch['supervisions']['start_sample'][seq_idx == i][j].item() for j in sort_orders[i]]
        for i in range(seq_idx.max()+1)
    ]
    num_frames_init = [
        [batch['supervisions']['num_samples'][seq_idx == i][j].item() for j in sort_orders[i]]
        for i in range(seq_idx.max()+1)
    ]

    with torch.inference_mode():
        start = time.time()
        x_lens, ctc_outputs_ = model.forward_target_speaker(
            feature,
            feature_lens,
            speakers=target_spks, 
        )
        end_nnet = time.time()
        subsampling_factor = params.subsampling_factor
        use_double_scores = params.use_double_scores
    
    ctc_outputs = [c.clone() for c in ctc_outputs_]
    for c in ctc_outputs_:
        del c
    del feature
    torch.cuda.empty_cache()
    for tgtspk in target_spks:
        logging.info(f"tgtspk: {tgtspk}")
        # Decoding with WFST supervisions
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

        ctc_outputs[tgtspk][..., 0] -= params.blank_weight
        
        lattice = get_lattice(
            nnet_output=ctc_outputs[tgtspk] * params.acoustic_weight,
            decoding_graph=graph,
            supervision_segments=supervision_segments,
            search_beam=params.search_beam,
            output_beam=params.output_beam,
            min_active_states=params.min_active_states,
            max_active_states=params.max_active_states,
            subsampling_factor=params.subsampling_factor,
        )
        
        if params.rescore:
            lm_scale_list = [0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05]
            #best_path_dict = rescore_with_whole_lattice(
            #    lattice=lattice,
            #    G_with_epsilon_loops=rescore,
            #    lm_scale_list=lm_scale_list,
            #)
            #best_path_dict = rescore_with_n_best_list(
            if x_lens[0] < 800: 
                best_path_dict = nbest_rescore_with_LM(
                    lattice=lattice,
                    LM=rescore,
                    num_paths=params.num_paths,
                    lm_scale_list=lm_scale_list,
                    nbest_scale=0.7,
                )
            else:
                logging.info(f"len: {x_lens[0]}. Falling back ...")
                one_best_decode = one_best_decoding(
                    lattice=lattice,
                    use_double_scores=params.use_double_scores
                )

                best_path_dict = {
                    "lm_scale_f{scale}": one_best_decode
                    for scale in lm_scale_list
                }
        else:
            best_path_dict = {
                "lm_scale_1.0": one_best_decoding(
                    lattice=lattice, use_double_scores=params.use_double_scores
                )
            }
       
        for lm_scale_str, best_path in best_path_dict.items():
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
            
            spks = []
            for i in range(units.shape.dim0):
                spks_ = []
                for j in units[i].view(-1):
                    spks_.append(tgtspk)
                spks.append(torch.LongTensor(spks_))

            
            cut_ids = [c.id for c in batch["supervisions"]["cut"]]
            tokens = {}
            for i, (cut_id, s, u, t) in enumerate(zip(cut_ids, spks, units, times)):
                spk2int = {k: j for j, k in enumerate(dict.fromkeys(speakers[i]))}
                int2spk = {j: k for k, j in spk2int.items()}
                speaker_hyps = {}
                speaker_refs = {}
                s_i = tgtspk
                if s_i in s:
                    speaker_hyps[s_i] = merge_bpe_with_times(
                        graph_compiler.sp,
                        u.view(-1)[s == s_i],
                        [t_.item() for t_ in t[s == s_i]],
                        (x_lens[i]+1)*params.frame_duration
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
                yield lm_scale_str, stm_hyp, stm_ref


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    graph_compiler: MDCTCGraphCompiler,
    graph,
    G,
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
    lm_scales = {}
    hyps, refs = [], []
    num_cuts = 0
    for batch_idx, batch in tqdm(enumerate(dl)):
        texts = batch["supervisions"]["text"]
        num_cuts += batch["inputs"].size(0)
        for lm_scale_str, hyps_, refs_ in decode_one_batch(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                graph=graph,
                batch=batch,
                rescore=G,
            ):
            if lm_scale_str not in lm_scales:
                lm_scales[lm_scale_str] = {'hyps': [], 'refs': []}
            lm_scales[lm_scale_str]['hyps'].extend(hyps_)
            lm_scales[lm_scale_str]['refs'].extend(refs_)
        batch_str = f"{batch_idx}/{num_batches}"
        logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    for lm_scale in lm_scales:
        lm_scales[lm_scale]['hyps'] = sorted(
            lm_scales[lm_scale]['hyps'],
            key=lambda x: (x["session_id"], x["start_time"])
        )
        lm_scales[lm_scale]['refs'] = sorted(
            lm_scales[lm_scale]['refs'],
            key=lambda x: (x["session_id"], x["start_time"])
        )
    return lm_scales



@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)
    args.lm_dir = Path(args.lm_dir)

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

    synth_cuts = librispeech.synth_cuts()
    l2m_test_both = librispeech.libri2mix_test_both_cuts()
    l2m_test_clean = librispeech.libri2mix_test_clean_cuts()
    l3m_test_clean = librispeech.libri3mix_test_clean_cuts()
    ls_test_other, ls_test_clean = librispeech.single_speaker_cuts()
    libricss = librispeech.libricss_cuts()
    ami_dev = librispeech.ami_dev()
    lsm_2 = librispeech.librispeechmix_2_cuts()
    lsm_3 = librispeech.librispeechmix_3_cuts()

    synth_dl = librispeech.valid_dataloaders(synth_cuts)
    l2m_test_both_dl = librispeech.valid_dataloaders(l2m_test_both)
    l2m_test_clean_dl = librispeech.valid_dataloaders(l2m_test_clean)
    l3m_test_clean_dl = librispeech.valid_dataloaders(l3m_test_clean)
    ls_test_other_dl = librispeech.valid_dataloaders(ls_test_other)
    ls_test_clean_dl = librispeech.valid_dataloaders(ls_test_clean)
    lsm_2_dl = librispeech.valid_dataloaders(lsm_2)
    lsm_3_dl = librispeech.valid_dataloaders(lsm_3)

    lcss_dl = librispeech.valid_dataloaders(libricss)
    ami_dev_dl = librispeech.valid_dataloaders(ami_dev)


    test_sets = ["synth", "libri3mix", "l2m_test_clean", "l2m_test_both", "lsm_2", "lsm_3", "ls_test_other", "ls_test_clean", "lcss", "ami_dev"]
    test_dl = [synth_dl, l3m_test_clean_dl, l2m_test_clean_dl, l2m_test_both_dl, lsm_2_dl, lsm_3_dl, ls_test_other_dl, ls_test_clean_dl, lcss_dl, ami_dev_dl]
    test_sets_dict = dict(zip(test_sets, test_dl))
    if params.test_sets is not None:
        test_sets = params.test_sets.split() 
    test_dl = [test_sets_dict[t] for t in test_sets]

    logging.info("Getting Decoding (HLG) graph ...")
    if not params.modified:
        HLG = k2.Fsa.from_dict(
            torch.load(f"{params.lang_dir}/HLG.pt", map_location=device, weights_only=False)
        )
    else:
        HLG = k2.Fsa.from_dict(
            torch.load(f"{params.lang_dir}/HLG_modified.pt", map_location=device, weights_only=False)
        )

    assert HLG.requires_grad is False

    if not hasattr(HLG, "lm_scores"):
        HLG.lm_scores = HLG.scores.clone()

    if params.rescore:
        if not (params.lm_dir / "G_4_gram.pt").is_file():
            logging.info("Loading G_4_gram.fst.txt")
            logging.warning("It may take 8 minutes.")
            with open(params.lm_dir / "G_4_gram.fst.txt") as f:
                lexicon = Lexicon(params.lang_dir)
                max_token_id = max(lexicon.tokens)
                num_classes = max_token_id + 1  # +1 for the blank
                params.vocab_size = num_classes

                first_word_disambig_id = lexicon.word_table["#0"]

                G = k2.Fsa.from_openfst(f.read(), acceptor=False)
                # G.aux_labels is not needed in later computations, so
                # remove it here.
                del G.aux_labels
                # CAUTION: The following line is crucial.
                # Arcs entering the back-off state have label equal to #0.
                # We have to change it to 0 here.
                G.labels[G.labels >= first_word_disambig_id] = 0
                # See https://github.com/k2-fsa/k2/issues/874
                # for why we need to set G.properties to None
                G.__dict__["_properties"] = None
                G = k2.Fsa.from_fsas([G]).to(device)
                G = k2.arc_sort(G)
                # Save a dummy value so that it can be loaded in C++.
                # See https://github.com/pytorch/pytorch/issues/67902
                # for why we need to do this.
                G.dummy = 1

                torch.save(G.as_dict(), params.lm_dir / "G_4_gram.pt")
        else:
            logging.info("Loading pre-compiled G_4_gram.pt")
            d = torch.load(params.lm_dir / "G_4_gram.pt", map_location=device)
            G = k2.Fsa.from_dict(d)
        
        # Add epsilon self-loops to G as we will compose
        # it with the whole lattice later
        
        #G = k2.add_epsilon_self_loops(G)
        #G = k2.arc_sort(G)
        #G = G.to(device)

        # G.lm_scores is used to replace HLG.lm_scores during
        # LM rescoring.
        G.lm_scores = G.scores.clone()
    else:
        G = None

    for set, dl in zip(test_sets, test_dl):
        lm_scales = decode_dataset(
            dl=dl,
            params=params,
            model=model,
            graph_compiler=graph_compiler,
            graph=HLG,
            G=G,
        )
    
        decode_dir = params.exp_dir / "decode"
        decode_dir.mkdir(parents=True, exist_ok=True)
        for lm_scale in lm_scales:
            hyps = lm_scales[lm_scale]['hyps']
            refs = lm_scales[lm_scale]['refs']
            with open(decode_dir / f"hyps_chkpt{params.iter}_avg{params.avg}_{lm_scale}_{set}_{params.suffix}.stm", "w") as f:
                for l in hyps:
                    print(f"{l['session_id']} 1 {l['speaker']} {l['start_time']} {l['end_time']} {l['words']}", file=f)
            
            with open(decode_dir / f"refs_chkpt{params.iter}_avg{params.avg}_{lm_scale}_{set}_{params.suffix}.stm", "w") as f:
                for l in refs:
                    print(f"{l['session_id']} 1 {l['speaker']} {l['start_time']} {l['end_time']} {l['words']}", file=f)

    logging.info("Done!")


if __name__ == "__main__":
    main()
