import math
from typing import List
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from .layers import Snake1d
from .layers import WNConv1d
from .layers import WNConvTranspose1d


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=1,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
            self,
            input_channel,
            channels,
            rates,
            d_out: int = 1,
            add_residual_connections: bool = False,
            residual_num_channels: Optional[list[int]] = None,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        self.add_residual_connections = add_residual_connections
        self.residual_num_channels = residual_num_channels

        if self.add_residual_connections and self.residual_num_channels is None:
            raise Exception('Num of residual channels cannot be 0.')

        if self.add_residual_connections:
            self.residual_combination_convs = []

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2 ** i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]
            if self.add_residual_connections:
                self.residual_combination_convs.append(WNConv1d(residual_num_channels[i] + input_dim, input_dim,
                                                                kernel_size=1, padding=0))

        if self.add_residual_connections:
            self.residual_combination_convs = nn.ModuleList(self.residual_combination_convs)

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_connections: Optional[list[torch.Tensor]] = None):
        if skip_connections is None:
            return self.model(x)

        skip_conn_idx = 0
        for layer in self.model:
            if isinstance(layer, DecoderBlock):
                skip = skip_connections[-(skip_conn_idx+1)]
                if self.add_residual_connections:
                    if skip.shape[-1] < x.shape[-1]:
                        x = x[..., :skip.shape[-1]]
                    else:
                        skip = skip[..., :x.shape[-1]]
                    x = torch.cat([x, skip], dim=1)
                    x = self.residual_combination_convs[skip_conn_idx](x)
                    skip_conn_idx += 1
            x = layer(x)
        return x


class DAC(BaseModel, CodecMixin):
    def __init__(
            self,
            encoder_dim: int = 64,
            encoder_rates: List[int] = (2, 4, 8, 8),
            latent_dim: int = None,
            decoder_dim: int = 1536,
            decoder_rates: List[int] = (8, 8, 4, 2),
            n_codebooks: int = 9,
            codebook_size: int = 1024,
            codebook_dim: Union[int, list] = 8,
            quantizer_dropout: bool = False,
            sample_rate: int = 44100,
            add_residual_connections: bool = False,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.add_residual_connections = add_residual_connections
        self.residual_res = None

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim, return_residuals=add_residual_connections)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            add_residual_connections=add_residual_connections,
            residual_num_channels=[(encoder_dim*(2**(i+1))) for i in range(len(decoder_rates)-1, -1, -1)],
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
            self,
            audio_data: torch.Tensor,
            n_quantizers: int = None,
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        if self.add_residual_connections:
            z, residual_res = self.encoder(audio_data, return_layer_outputs=self.add_residual_connections)
            self.residual_res = residual_res
        else:
            z = self.encoder(audio_data)

        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers
        )
        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        if self.add_residual_connections:
            return self.decoder(z, skip_connections=self.residual_res)

        return self.decoder(z)

    def forward(
            self,
            audio_data: torch.Tensor,
            sample_rate: int = None,
            n_quantizers: int = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers
        )

        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }
