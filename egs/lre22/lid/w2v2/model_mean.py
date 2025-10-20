from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import Wav2Vec2Model


class AveragePool(nn.Module):
    def __init__(self):
        super(AveragePool, self).__init__()

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        # Create mask
        max_seq_length = x_lens.max().item()
        # Step 2: Create a binary mask
        mask = torch.arange(max_seq_length)[None, :].to(x.device) >= x_lens[:, None]
        x[mask] = torch.nan
        return x.nanmean(dim=1), None 


class LIDModel(nn.Module):
    def __init__(self,
        num_classes: int,
        pooling_type: str = "pre",
        modelpath: str = "facebook/mms-300m",
        cache_dir: str = "/expscratch/mwiesner/testthis",
        freeze_feat_extractor: bool = True,
    ):
        super(LIDModel, self).__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(
            modelpath, cache_dir=cache_dir
        )
        self.odim = self._get_output_dim()
        self.frozen = False
        self.freeze_feat_extractor = freeze_feat_extractor
        self.pooling = AveragePool()
        self.pooling_type = pooling_type
        self.linear_out = nn.Linear(self.odim, num_classes)
    
    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x.squeeze(-1), output_hidden_states=False)[0]
        
        for width, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            x_lens = torch.floor((x_lens - width) / stride + 1)

        if self.pooling_type == "pre":
            x, w = self.pooling(x, x_lens)
            x = self.linear_out(x)
        elif self.pooling_type == "post":
            x = self.linear_out(x)
            x, w = self.pooling(x, x_lens)
        return x, w
    
    def get_global_embeddings(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> torch.Tensor:
        x = self.encoder(
            x.squeeze(-1), output_hidden_states=False
        )[0]
        
        for width, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            x_lens = torch.floor((x_lens - width) / stride + 1)

        if self.pooling_type == "pre":
            x, w = self.pooling(x, x_lens)
            return x
        elif self.pooling_type == "post":
            x = self.linear_out(x)
            x, w = self.pooling(x, x_lens)
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


        
