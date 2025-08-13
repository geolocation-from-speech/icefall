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

from typing import List, Tuple

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from transformer import encoder_padding_mask

from mdctc_graph_compiler import MDCTCGraphCompiler
from icefall.utils import encode_supervisions



class AttentionPool(nn.Module):
    """
        This class implements an alternative to mean pooling that seemed to
        perform slightly better for the geolocation task. It takes an attention
        layer and a learned embedding as a query representing the task of
        geolocation and uses them to pool the data. The advantage is that
        the attention weights are interprettable so it can help in debugging to
        inspect on what parts of the speech the model is learning to geolocate
        
        :param att: The attention layer to be used in attention pooling
        :param query_embed: The embedding (just a vector) representing the task
            of geolocating speech.
        :return: The attention pooling module
    """
    def __init__(self, att, query_embed):
        super(AttentionPool, self).__init__()
        self.query_embed = query_embed
        self.att = att

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            The forward function of this module.

            :param x: The input tensor to be pooled (B x T x D)
            :param x_lens: The tensor of lengths corresponding to each element
                in the first dimension of x.
            :return: The tuple of tensors representing the pooled data as well
                as the corresponding attention weights.
            :rtype: Tuple[torch.Tensor, torch.Tensor] (B x D)
        """ 
        # Create mask
        max_seq_length = x_lens.max().item()

        # Step 2: Create a binary mask
        mask = torch.arange(max_seq_length)[None, :].to(x.device) >= x_lens[:, None]
        
        # Step 3: Expand the mask to match the shape required by MultiheadAttention
        # The mask should have shape (batch_size, 1, 1, max_seq_length)
        x, w = self.att(
            self.query_embed.unsqueeze(0).unsqueeze(1).repeat(x.size(0), 1, 1),
            x,
            x,
            key_padding_mask=mask
        )
        x = x.squeeze(1)
        return x, w



class LinearUpsampler(nn.Module):
    def __init__(self, dim, num_reps):
        super().__init__()
        self.linear = nn.Linear(dim, num_reps*dim)
        self.dim = dim
        self.num_reps = num_reps

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        D = x.size(-1)
        B = x.size(0)
        x = self.linear(x)
        return x.reshape(B, self.num_reps * L, D), x_lens * self.num_reps


class GatedUpsampler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim 
        self.scale = nn.Parameter(
            torch.FloatTensor(self.dim).uniform_(-1, 1)
        )

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        #weights = nn.functional.softmax(self.scale, -1)
        weights = nn.functional.sigmoid(self.scale)
        output = torch.stack(
            [x * weights, x * (1-weights)], dim=1
        ).reshape(x.size(0), -1, x.size(-1)).to(x.device)
        return output, x_lens * 2
    

class MDCTCModel(nn.Module):
    """It implements an MDCTC model with an auxiliary attention head."""

    def __init__(
        self,
        encoder: EncoderInterface,
        encoder_dim: int,
        vocab_size: int,
        freeze_feat_extractor: bool = True,
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
        # Freeze the feature extractor (CNN frontend)
        try:
            for param in encoder.feature_extractor.parameters():
                param.requires_grad = False
        except AttributeError:
            for param in encoder.model.feature_extractor.parameters():
                param.requires_grad = False
 
        # value
        # Adding in transposed convolution
        #self.upsampler = nn.ConvTranspose1d(
        #    encoder_dim,
        #    encoder_dim,
        #    kernel_size,
        #    stride=stride,
        #    padding=(kernel_size-stride)//2
        #)

        #self.upsampler = GatedUpsampler(encoder_dim)
        #self.upsampler = LinearUpsampler(encoder_dim, stride) 
        self.freeze_feat_extractor = freeze_feat_extractor
        self.frozen = False
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
        assert x_lens.ndim == 1, x_lens.shape
        nnet_output = self.encoder(x)[0]
        
        for width, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            x_lens = torch.floor((x_lens - width) / stride + 1)

        assert torch.all(x_lens > 0)
        
        # We will upsample here to handle overlapping speech
        #nnet_output, x_lens = self.upsampler(nnet_output, x_lens)
        #nnet_output = self.upsampler(
        #    nnet_output.transpose(1, 2)
        #).transpose(1, 2) 
        #x_lens = x_lens * 2

        # compute ctc log-probs
        nnet_output = nnet_output.transpose(1, 2)
        nnet_output = nn.functional.avg_pool1d(nnet_output, kernel_size=2, stride=2)
        nnet_output = nnet_output.transpose(1, 2)
        x_lens = torch.floor((x_lens - 2) / 2 + 1).to(torch.int32)
        ctc_output = self.ctc_output(nnet_output)

        return ctc_output, x_lens

    def freeze_encoder(self):
        try:
            for p in self.encoder.encoder.parameters():
                if p.requires_grad:
                    p.requires_grad = False
            self.frozen = True
        except AttributeError:
            for p in self.encoder.model.parameters():
                if p.requires_grad:
                    p.requires_grad = False
            self.frozen = True


    def unfreeze_encoder(self):
        try:
            for i, p in enumerate(self.encoder.encoder.parameters()):
                p.requires_grad = True
            if self.freeze_feat_extractor:
                # Freeze the feature extractor (CNN frontend)
                for param in self.encoder.feature_extractor.parameters():
                    param.requires_grad = False
            self.frozen = False
        except AttributeError:
            for i, p in enumerate(self.encoder.model.parameters()):
                p.requires_grad = True
            if self.freeze_feat_extractor:
                # Freeze the feature extractor (CNN frontend)
                for param in self.encoder.model.feature_extractor.parameters():
                    param.requires_grad = False
            self.frozen = False

