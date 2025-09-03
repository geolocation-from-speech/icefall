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


class MDCTCModel(nn.Module):
    """It implements an MDCTC model with an auxiliary attention head."""

    def __init__(
        self,
        encoder,
        encoder_dim: int,
        vocab_size1: int,
        vocab_size2: int,
        freeze_feat_extractor: bool = True,
        hat: bool = False,
        layer_norm: bool = False,
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

        self.freeze_feat_extractor = freeze_feat_extractor
        self.frozen = False
        self.encoder = encoder
        self.ctc_output1 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(encoder_dim, vocab_size1),
        ) 

        self.ctc_output2 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(encoder_dim, vocab_size2),
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.hat = hat
        self.use_layer_norm = layer_norm

    @torch.jit.ignore
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        speaker_mask = None,
    ) -> torch.Tensor:
        """
        Args:
          x:
            Tensor of dimension (N, T, C) where N is the batch size,
            T is the number of frames, and C is the feature dimension.
          x_lens:
            Tensor of dimension (N,) where N is the batch size.
          speaker_mask:
            mask out specific speakers
        Returns:
          Return the CTC loss, attention loss, and the total number of frames.
        """
        assert x_lens.ndim == 1, x_lens.shape
        nnet_output = self.encoder(x)[0]
        
        for width, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            x_lens = torch.floor((x_lens - width) / stride + 1)

        assert torch.all(x_lens > 0)
        
        # compute ctc log-probs
        nnet_output = nnet_output.transpose(1, 2)
        nnet_output = nn.functional.avg_pool1d(nnet_output, kernel_size=2, stride=2)
        nnet_output = nnet_output.transpose(1, 2)
        x_lens = torch.floor((x_lens - 2) / 2 + 1).to(torch.int32)
        if self.use_layer_norm:
            nnet_output = self.layer_norm(nnet_output)
        ctc_output1 = self.ctc_output1(nnet_output)
        ctc_output2 = self.ctc_output2(nnet_output)
        if self.hat:
            out1 = self.log_softmax(ctc_output1)
            blank = out1[..., 0]
            out2 = self.log_softmax(ctc_output2)
            if speaker_mask is not None:
                out2[..., speaker_mask] = -torch.inf
            out = out1[..., 1:].unsqueeze(-1) + out2.unsqueeze(-2)
            out = out.permute(0, 1, 3, 2)
            out = out.reshape(out.size(0), out.size(1), -1)
            out = torch.cat([blank.unsqueeze(-1), out], dim=-1)
        else:
            blank = ctc_output1[..., 0]
            out = ctc_output1[..., 1:].unsqueeze(-1) + ctc_output2.unsqueeze(-2)
            out = out.permute(0, 1, 3, 2)
            out = out.reshape(out.size(0), out.size(1), -1)
            out = torch.cat([blank.unsqueeze(-1), out], dim=-1)
            out = self.log_softmax(out)
        return out, x_lens

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

