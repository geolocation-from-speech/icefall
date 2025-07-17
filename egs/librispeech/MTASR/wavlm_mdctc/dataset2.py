#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Author: Matthew Wiesner
# Year: 2025

from collections import defaultdict
from typing import Callable, Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import LOG_EPSILON, compute_num_frames, ifnone
from lhotse.workarounds import Hdf5MemoryIssueFix
import re
from itertools import groupby


class K2MultiTalkerSpeechRecognitionDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the multi-talker ASR task using k2 library.
    We support the modeling framework known as Streaming Unmixing and Recognition
    Transducer (SURT), as described in [1] and [2], but this dataset can also be
    used for other multi-talker ASR approaches such as MT-RNNT [3] and SOT [4].
    See icefall recipe for usage: https://github.com/k2-fsa/icefall/pull/1126.

    We take a cut containing possibly overlapping speech and split the supervision
    segments into one of N channels based on their start times (N is provided), known
    as ``heuristic error assignment training`` (HEAT) [1]. The supervision segments
    in each channel are then concatenated and used as the supervision for that channel.
    If we have features for the source cuts, we can also return them for use in masking
    losses, for instance.

    [1] Lu, L., Kanda, N., Li, J., & Gong, Y. (2021). Streaming end-to-end multi-talker
    speech recognition. IEEE Signal Processing Letters, 28, 803-807.

    [2] Raj, D., Lu, L., Chen, Z., Gaur, Y., & Li, J. (2022, May). Continuous streaming
    multi-talker asr with dual-path transducers. In ICASSP 2022-2022 IEEE International
    Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 7317-7321). IEEE.

    [3] Sklyar, I., Piunova, A., Zheng, X., & Liu, Y. (2022, May). Multi-turn RNN-T for
    streaming recognition of multi-party speech. In ICASSP 2022-2022 IEEE International
    Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 8402-8406). IEEE.

    [4] Kanda, N., Gaur, Y., Wang, X., Meng, Z., & Yoshioka, T. (2020). Serialized output
    training for end-to-end overlapped speech recognition. arXiv preprint arXiv:2003.12687.

    .. hint:: Training mixtures can be simulated from single-speaker utterances using
        :class:`~lhotse.workflows.MeetingSimulation` workflow.

    This dataset expects to be queried with lists of cut IDs,
    for which it loads features and automatically collates/batches them.

    To use it with a PyTorch DataLoader, set ``batch_size=None``
    and provide a :class:`SimpleCutSampler` sampler.

    Each item in this dataset is a dict of:

    .. code-block::

        {
            'inputs': float tensor with shape determined by :attr:`input_strategy`:
                      - single-channel:
                        - features: (B, T, F)
                        - audio: (B, T)
                      - multi-channel: currently not supported
            'input_lens': int tensor of shape (B,)
            'supervisions': list of lists of supervision segments, where the outer list is
                        batch, and the inner list is indexed by channel. So ``len(supervisions) == B``,
                        and ``len(supervisions[i]) == num_channels``. Note that some channels may
                        have no supervision segments.
            'text': list of lists of strings, where the outer list is batch, and the inner list
                    is indexed by channel. So ``len(text) == B``, and ``len(text[i]) == num_channels``.
                    Each element contains the text of the supervision segments in that channel,
                    joined by the :attr:`text_delimiter`. Note that some channels may have no
                    supervision segments, so the corresponding text will be an empty string.
        }

    If ``return_cuts`` is ``True``, each item will also contain a ``cuts`` field with the
    :class:`~lhotse.cut.Cut` objects used to create the batch.

    Additionally, if ``return_sources`` is ``True``, each item will contain:

    .. code-block::

        {
            'source_feats': list of list of float tensors. The outer list is batch, and the inner
                            list is number of segments in the mixture. Each element denotes
                            the features of the source cut in the mixture.
            'source_boundaries': list of list of tuples, where the outer list is batch, and the inner list
                                is number of segments in the mixture. Each element denotes
                                the start and end frame of the source cut in the mixture.
        }

    In order to return the source features and boundaries, we expect the cuts to contain
    some additional fields:

    - ``source_feats``: a float tensor representing the features of all the source segments,
        concatenated together along T dimension, in order of their start times.
    - ``source_feat_offsets``: a list of ints, where each element denotes the offset of the
        source segments in the ``source_feats`` tensor.

    See https://github.com/lhotse-speech/lhotse/discussions/1008#discussioncomment-5511746
    for example code to create these fields.


    Dimension symbols legend:
    * ``B`` - batch size (number of Cuts)
    * ``S`` - number of supervision segments (greater or equal to B, as each Cut may have multiple supervisions)
    * ``T`` - number of frames of the longest Cut
    * ``F`` - number of features

    The 'sequence_idx' field is the index of the Cut used to create the example in the Dataset.
    """

    def __init__(
        self,
        graph_compiler,
        return_cuts: bool = False,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = PrecomputedFeatures(),
    ):
        """
        k2 ASR IterableDataset constructor.

        :param return_cuts: When ``True``, will additionally return a "cut" field in each batch with the Cut
            objects used to create that batch.
        :param cut_transforms: A list of transforms to be applied on each sampled batch,
            before converting cuts to an input representation (audio/features).
            Examples: cut concatenation, noise cuts mixing, etc.
        :param input_transforms: A list of transforms to be applied on each sampled batch,
            after the cuts are converted to audio/features.
            Examples: normalization, SpecAugment, etc.
        :param input_strategy: Converts cuts into a collated batch of audio/features.
            By default, reads pre-computed features from disk.
        """

        super().__init__()
        # Initialize the fields
        self.return_cuts = return_cuts
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy
        self.graph_compiler = graph_compiler 
        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Return a new batch, with the batch size automatically determined using the constraints
        of max_duration and max_cuts.
        """
        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        # Optional CutSet transforms - e.g. padding, or speed perturbation that adjusts
        # the supervision boundaries.
        for tnfm in self.cut_transforms:
            cuts = tnfm(cuts)

        # Sort the cuts again after transforms
        cuts = cuts.sort_by_duration(ascending=False)

        # Get a tensor with batched feature matrices, shape (B, T, F)
        # Collation performs auto-padding, if necessary.
        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            # An input strategy with fault tolerant audio reading mode.
            # "cuts" may be a subset of the original "cuts" variable,
            # that only has cuts for which we succesfully read the audio.
            inputs, _, cuts = input_tpl
        else:
            inputs, _ = input_tpl

        # Get a dict of tensors that encode the positional information about supervisions
        # in the batch of feature matrices. The tensors are named "sequence_idx",
        # "start_frame/sample" and "num_frames/samples".
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)
        
        # Get the number of frames per audio_cut
        seq_idx = supervision_intervals['sequence_idx']
        num_frames = []
        fs = self.input_strategy.extractor.frame_shift
        for c in cuts:
            nf = compute_num_frames(
                c.duration, frame_shift=fs, sampling_rate=c.sampling_rate
            )
            num_frames.append(min(inputs.size(1), nf))
        num_frames = torch.LongTensor(num_frames)
        
        # Apply all available transforms on the inputs, i.e. either audio or features.
        # This could be feature extraction, global MVN, SpecAugment, etc.
        segments = torch.stack(list(supervision_intervals.values()), dim=1)
        for tnfm in self.input_transforms:
            inputs = tnfm(inputs, supervision_segments=segments)

        
        texts = extract_texts_from(cuts)
        seq_idx = supervision_intervals['sequence_idx']
        start_frames = [
            supervision_intervals['start_frame'][seq_idx == i].tolist()
            for i in range(seq_idx.max()+1)
        ]
        num_frames_init = [
            supervision_intervals['num_frames'][seq_idx == i].tolist()
            for i in range(seq_idx.max()+1)
        ]

        decoding_graphs = self.graph_compiler.compile(
            texts, start_frames, num_frames_init
        )

        batch = {
            "inputs": inputs,
            "supervisions": default_collate(
                [
                    {
                        "text": supervision.text,
                    }
                    for sequence_idx, cut in enumerate(cuts)
                    for supervision in cut.supervisions
                ]
            ),
            "texts": texts,
            "num_frames": num_frames,
            "graphs": decoding_graphs,
        }
        # Update the 'supervisions' field with sequence_idx and start/num frames/samples
        batch["supervisions"].update(supervision_intervals)
        if self.return_cuts:
            batch["supervisions"]["cut"] = [
                cut for cut in cuts for sup in cut.supervisions
            ]

        has_word_alignments = all(
            s.alignment is not None and "word" in s.alignment
            for c in cuts
            for s in c.supervisions
        )
        if has_word_alignments:
            # TODO: might need to refactor BatchIO API to move the following conditional logic
            #       into these objects (e.g. use like: self.input_strategy.convert_timestamp(),
            #       that returns either num_frames or num_samples depending on the strategy).
            words, starts, ends = [], [], []
            frame_shift = cuts[0].frame_shift
            sampling_rate = cuts[0].sampling_rate
            if frame_shift is None:
                try:
                    frame_shift = self.input_strategy.extractor.frame_shift
                except AttributeError:
                    raise ValueError(
                        "Can't determine the frame_shift -- it is not present either in cuts or the input_strategy. "
                    )
            for c in cuts:
                for s in c.supervisions:
                    words.append([aliword.symbol for aliword in s.alignment["word"]])
                    starts.append(
                        [
                            compute_num_frames(
                                aliword.start,
                                frame_shift=frame_shift,
                                sampling_rate=sampling_rate,
                            )
                            for aliword in s.alignment["word"]
                        ]
                    )
                    ends.append(
                        [
                            compute_num_frames(
                                aliword.end,
                                frame_shift=frame_shift,
                                sampling_rate=sampling_rate,
                            )
                            for aliword in s.alignment["word"]
                        ]
                    )
            batch["supervisions"]["word"] = words
            batch["supervisions"]["word_start"] = starts
            batch["supervisions"]["word_end"] = ends

        return batch


def validate_for_asr(cuts: CutSet) -> None:
    validate(cuts)
    tol = 2e-3  # 1ms
    for cut in cuts:
        for supervision in cut.supervisions:
            assert supervision.start >= -tol, (
                f"Supervisions starting before the cut are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )
            assert supervision.duration <= cut.duration + tol, (
                f"Supervisions ending after the cut "
                f"are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )


def extract_texts_from(cuts):
    texts = []
    for c in cuts:
        groups = groupby(
            sorted(c.supervisions, key=lambda x: 'noise' if x.speaker is None else x.speaker),
            lambda x: 'noise' if x.speaker is None else x.speaker
        )
        cut_texts = []
        for k, g in groups:
            g_ = list(g)
            text_ = " ".join([g_i.text for g_i in g_])
            text_ = re.sub(r" +", " ", text_)
            text_ = re.sub(r"^ ", "", text_)
            cut_texts.append(text_)
        texts.append(cut_texts)
    return texts
