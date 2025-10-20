# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
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


from transformers import Wav2Vec2Model
from typing import Optional, Tuple

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface

from zipformer import Zipformer2
from icefall.utils import add_sos, make_pad_mask
from scaling import ScaledLinear


class AsrModel(nn.Module):
    def __init__(
        self,
        modelpath: str = 'facebook/mms-300m',
        vocab_size: int = 10000,
        feat_att_dim: int = 128,
    ):
        """
        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
            It is used when use_transducer is True.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
            It is used when use_transducer is True.
          use_transducer:
            Whether use transducer head. Default: True.
          use_ctc:
            Whether use CTC head. Default: False.
        """
        super().__init__()

        self.encoder_embed = Wav2Vec2Model.from_pretrained(modelpath)

        self.encoder_embed.feature_extractor._freeze_parameters()
        for p in self.encoder_embed.parameters():
            p.requires_grad = False

        self.embed_dim = self._get_output_dim()
        self.embed_proj = nn.Linear(self.embed_dim, 192) 
        self.num_layers = self.encoder_embed.config.num_hidden_layers + 1  
        
        self.feature_attention = nn.Sequential(
            nn.Linear(self.num_layers, feat_att_dim),
            nn.ReLU(),
            nn.Linear(feat_att_dim, self.num_layers),
            nn.Softmax(dim=-1),
        )

        self.encoder = Zipformer2(
            output_downsampling_factor=2,
            downsampling_factor=(1, 2, 4, 8, 4, 2),
            num_encoder_layers=(2, 2, 3, 4, 3, 2),
            encoder_dim=(192, 256, 384, 512, 384, 256),
            encoder_unmasked_dim=(192, 192, 256, 256, 256, 192),
            query_head_dim=32,
            pos_head_dim=4,
            value_head_dim=12,
            pos_dim=48,
            num_heads=(4, 4, 4, 8, 4, 4),
            feedforward_dim=(512, 768, 1024, 1536, 1024, 768),
            cnn_module_kernel=(31, 31, 15, 15, 15, 31)
        )

        # The output layer
        self.ctc_output = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(512, vocab_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
            :param x: The tensor of (raw) input audio
            :param x_lens: The tensor containing lengths for each element in x
            :return: A tuple of the cartesian coordinates on the unit sphere
                modeling the surface of the Earth and the pooling weights. 
        """
        with torch.set_grad_enabled(False):
            outputs = self.encoder_embed(
                x.squeeze(-1), output_hidden_states=True
            )
       
        # B x L x D x num_layers
        hidden_states = torch.stack(outputs.hidden_states, dim=-1)

        # use the global mean across dimension as key
        mean_pooled = hidden_states.mean(dim=2) # B x L x num_layers
        
        # Internally, this gets converted to B x num_layers x num_layers
        weights = self.feature_attention(mean_pooled)
       
        # Apply weights
        x = (hidden_states * weights.unsqueeze(2)).sum(dim=-1)
         
        # For all the Wav2Vec2.0 models, the down-sampling happens in the
        # convolutional layers, which always have the same structure. We can
        # just hardcode the stride, kernel width, and dilation of each layer
        # since for this class of model it will always be the same.
        for width, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            x_lens = torch.floor((x_lens - width) / stride + 1)
        x_lens = x_lens.to(torch.int32)
        
        x = self.embed_proj(x)
        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        x, x_lens = self.encoder(x, x_lens, src_key_padding_mask)
        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        x = self.ctc_output(x)
        return x, x_lens

    #def forward_ctc(
    #    self,
    #    x: torch.Tensor,
    #    x_lens: torch.Tensor,
    #    targets: k2.RaggedTensor,
    #    target_lengths: torch.Tensor,
    #) -> torch.Tensor:
    #    targets = targets.values
    #    ctc_loss = torch.nn.functional.ctc_loss(
    #        log_probs=x.permute(1, 0, 2),
    #        targets=targets,
    #        input_lengths=x_lens,
    #        target_lengths=target_lengths,
    #        reduction="sum",
    #    )
    #    return ctc_loss

    def _get_output_dim(self):
        x = torch.rand(1, 400)
        return self.encoder_embed(x).last_hidden_state.size(-1)

        
