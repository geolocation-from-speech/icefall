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
       
        return x, x_lens

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

 
