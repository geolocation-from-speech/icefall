# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
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

from typing import List

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from transformer import encoder_padding_mask

from mdctc_graph_compiler import MDCTCGraphCompiler
from icefall.utils import encode_supervisions


class MDCTCModel(nn.Module):
    """It implements an MDCTC model with an auxiliary attention head."""

    def __init__(
        self,
        encoder: EncoderInterface,
        encoder_dim: int,
        vocab_size: int,
        kernel_size: int = 2,
        stride: int = 2,
    ):
        """
        Args:
          encoder:
            An instance of `EncoderInterface`. The shared encoder for the CTC and attention
            branches
          encoder_dim:
            Dimension of the encoder output.
          vocab_size:
            Number of tokens of the modeling unit including blank.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)

        # This assertion is to make sure that we upsample by a whole integer
        # value
        # Lo = (Li−1)×stride − 2×padding + dilation×(kernel_size−1) + output_padding + 1
        assert (kernel_size - stride) % 2 == 0
        # Adding in transposed convolution
        self.upsampler = nn.ConvTranspose1d(
            encoder_dim,
            encoder_dim,
            kernel_size,
            stride=stride,
            padding=(kernel_size-stride)//2
        )
        
        self.encoder = encoder
        self.ctc_output = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(encoder_dim, vocab_size),
            nn.LogSoftmax(dim=-1),
        )

    @torch.jit.ignore
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          x:
            Tensor of dimension (N, T, C) where N is the batch size,
            T is the number of frames, and C is the feature dimension.
          x_lens:
            Tensor of dimension (N,) where N is the batch size.
          texts:
            the training transcripts for the cut
          graph_compiler:
            It is used to compile a decoding graph from texts.
          subsampling_factor:
            It is used to compute the `supervisions` for the encoder.
          beam_size:
            Beam size used in `k2.ctc_loss`.
          reduction:
            Reduction method used in `k2.ctc_loss`.
          use_double_scores:
            If True, use double precision in `k2.ctc_loss`.
        Returns:
          Return the CTC loss, attention loss, and the total number of frames.
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        nnet_output, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)
        
        # We will upsample here to handle overlapping speech
        nnet_output = self.upsampler(
            nnet_output.transpose(1, 2)
        ).transpose(1, 2) 

        # compute ctc log-probs
        ctc_output = self.ctc_output(nnet_output)

        #preds = ctc_output.argmax(-1)
        #hyps = [
        #    preds[i].unique_consecutive()[preds[i].unique_consecutive != 0].squeeze(0).tolist()
        #    for i in range(preds.size(0))
        #]
        #try:
        #    print(f"hyp: {graph_compiler.sp.decode(hyps[0][0:20])}")
        #except:
        #    import pdb; pdb.set_trace()

        #for t_idx, text in enumerate(texts[0]):
        #    print(f"ref_{t_idx}: {text}")
        ## NOTE: We need `encode_supervisions` to sort sequences with
        ## different duration in decreasing order, required by
        ## `k2.intersect_dense` called in `k2.ctc_loss`
        ## This part is to get the supervision_segments
        #sequence_idx = torch.arange(
        #    0, x_lens.size(0),
        #).unsqueeze(0).t().to(torch.int32)

        #start_frame = torch.zeros(
        #    [x_lens.size(0)], dtype=torch.int32,
        #).unsqueeze(0).t()

        #num_frames = (x_lens * 2).unsqueeze(1).to(torch.int32).cpu()

        #supervision_segments = torch.cat(
        #    [sequence_idx, start_frame, num_frames],
        #    dim=1,
        #)
        #supervision_segments = supervision_segments.to(torch.int32)


        ## Works with a BPE model
        #decoding_graphs = graph_compiler.compile(texts)
        #decoding_graphs = decoding_graphs.to(x.device)

        #dense_fsa_vec = k2.DenseFsaVec(
        #    ctc_output,
        #    supervision_segments.cpu(),
        #    allow_truncate=subsampling_factor - 1,
        #)

        #ctc_loss = k2.ctc_loss(
        #    decoding_graph=decoding_graphs,
        #    dense_fsa_vec=dense_fsa_vec,
        #    output_beam=beam_size,
        #    reduction=reduction,
        #    use_double_scores=use_double_scores,
        #)

        #return loss, num_frames.sum().item()
        return ctc_output, x_lens
