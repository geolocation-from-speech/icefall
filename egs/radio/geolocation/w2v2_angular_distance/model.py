# Copyright    2023 Johns Hopkins University (Author: Matthew Wiesner, Patrick Foley)
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


from transformers import Wav2Vec2ForCTC, Wav2Vec2ForPreTraining
import torch
import torch.nn as nn
from typing import Optional, Tuple


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


class AveragePool(nn.Module):
    """
        A class for average pooling.
    """
    def __init__(self):
        super(AveragePool, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """
            The forward function of this module.

            :param x: The input tensor to be pooled (B x T x D)
            :param x_lens: The tensor of lengths corresponding to each element
                in the first dimension of x.
            :return: To be parallel with the AttentionPool module, it return
                a tuple of a tensor representing the pooled data and None.
            :rtype: Tuple[torch.Tensor, None] (B x D) 
        """ 

        # Create mask
        max_seq_length = x_lens.max().item()
        # Step 2: Create a binary mask
        mask = torch.arange(max_seq_length)[None, :].to(x.device) >= x_lens[:, None]
        x[mask] = torch.nan
        return x.nanmean(dim=1), None 


class Wav2Vec2Model(nn.Module):
    """
        This class is mostly a wrapper around a model you can just load from
        huggingface which is used as an encoder. A geolocation head is attached
        for the actual task of geolocation. Part of the geolocation head is
        the pooling mechanism.

        :param modelpath: The huggingface path to use to download the data
        :param freeze_feat_extractor: The Wav2Vec2.0 model feature extractor,
            i.e., the convolutional layers at the input, is normally
            frozen during model finetuning. The option allows you to either
            freeze those parameters or not during training.
        :param pooling_loc: An integer to select where the pooling will take
            place.
            0 --> Pooling takes place immediately after the encoder.
            1 --> Pooling takes place after transformation from hidden dimension
                into Cartesian coordinates, which do not necessarily lie on 
                the surface of the spherical model of the Earth.
            2 --> Pooling is after projection onto the surface of the Earth.
                  The correpsonds to the MLE estimate of the mean parameter in
                  the von Mises-Fisher distribution.
        :param pooling_type: A string to specify which pooling mechanism to use.
            "avg" --> AveragePooling
            "att" --> AttentionPooling
            
            Attention pooling should only really be used if pooling_loc = 0
    """
    def __init__(self,
        modelpath: str = 'facebook/mms-300m',
        freeze_feat_extractor: bool = True,
        pooling_loc: int = 0,
        pooling_type: str = 'att',
        pooling_heads: int = 1,
    ):
        super(Wav2Vec2Model, self).__init__()
        try:
            self.encoder = Wav2Vec2ForCTC.from_pretrained(modelpath).wav2vec2
        except:
            self.encoder = Wav2Vec2ForPreTraining.from_pretrained(modelpath).wav2vec2

        if freeze_feat_extractor:
            self.encoder.feature_extractor._freeze_parameters()
        self.freeze_feat_extractor = freeze_feat_extractor
        self.odim = self._get_output_dim()
        
        self.frozen = False
        if pooling_type == 'att':
            assert pooling_loc == 0, "pooling_type = 'att' can only be used with pooling_loc = 0"
            self.att = nn.MultiheadAttention(
                self.odim, pooling_heads, batch_first=True
            )
            self.loc_embed = nn.Parameter(
                torch.FloatTensor(self.odim).uniform_(-1, 1)
            )
            self.pooling = AttentionPool(self.att, self.loc_embed)
        elif pooling_type == 'avg':
            self.pooling = AveragePool()
        self.pooling_type = pooling_type
        # pooling loc is on 0: embeddings 1: unnormalized coords, 2: normalized coords
        self.pooling_loc = pooling_loc
        
        # The output layer
        self.linear_out = nn.Linear(self.odim, 3)

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
        x = self.encoder(
            x.squeeze(-1), output_hidden_states=False
        )[0]
       
        # For all the Wav2Vec2.0 models, the down-sampling happens in the
        # convolutional layers, which always have the same structure. We can
        # just hardcode the stride, kernel width, and dilation of each layer
        # since for this class of model it will always be the same.
        for width, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            x_lens = torch.floor((x_lens - width) / stride + 1)
       
        # Figure out how to pool
        if self.pooling_loc == 0: 
            x, w = self.pooling(x, x_lens)
            x = self.linear_out(x)
            x = x.div(x.norm(dim=1).unsqueeze(-1))
        elif self.pooling_loc == 1:
            x = self.linear_out(x)
            x, w = self.pooling(x, x_lens)
            x = x.div(x.norm(dim=1).unsqueeze(-1))
        elif self.pooling_loc == 2:
            x = self.linear_out(x)
            x = x.div(x.norm(dim=1).unsqueeze(-1))
            x, w = self.pooling(x, x_lens)
            x = x.div(x.norm(dim=1).unsqueeze(-1))
        return x, w

    
    def get_global_embeddings(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor
    ) -> torch.Tensor:
        x = self.encoder(
            x.squeeze(-1), output_hidden_states=False
        )[0]
        
        for width, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            x_lens = torch.floor((x_lens - width) / stride + 1)
        
        if self.pooling_loc == 0:
            x, _ = self.pooling(x, x_lens)
        elif self.pooling_loc == 1:
            x = self.linear_out(x)
            x, _ = self.pooling(x, x_lens)
        elif self.pooling_loc == 2:
            x = self.linear_out(x)
            x = x.div(x.norm(dim=1).unsqueeze(-1))
            x, _ = self.pooling(x, x_lens)
            x = x.div(x.norm(dim=1).unsqueeze(-1))
        return x
    
    def freeze_encoder(self):
        for p in self.encoder.encoder.parameters():
            if p.requires_grad:
                p.requires_grad = False
        self.frozen = True

    def unfreeze_encoder(self):
        for i, p in enumerate(self.encoder.encoder.parameters()):
            p.requires_grad = True
        if self.freeze_feat_extractor:
            self.encoder.feature_extractor._freeze_parameters()
        self.frozen = False

    def _get_output_dim(self):
        x = torch.rand(1, 400)
        return self.encoder(x).last_hidden_state.size(-1)

    # TODO: Use the attention weights to repurpose this model to do some sort of
    # VAD.
    #def forward_vad(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
    #    x = self.encoder(
    #        x.squeeze(-1), output_hidden_states=False
    #    )[0]
    #    
    #    for width, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
    #        x_lens = torch.floor((x_lens - width) / stride + 1)
    #        if self.pooling_loc == 1:
    #            x = self.linear_out(x)
    #            x, w = self.pooling(x, x_lens)

 
