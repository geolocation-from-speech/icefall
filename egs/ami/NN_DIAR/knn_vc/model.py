import math
from tqdm import tqdm
from pathlib import Path
from typing import List
from typing import Optional, Union, Generator
from torchaudio.pipelines import WAVLM_LARGE
from wavlm.WavLM import WavLM, WavLMConfig
from lhotse import CutSet
import torch.nn.functional as F
from time import time

import numpy as np
import torch
from torch import nn

import faiss

from layers import Snake1d
from layers import WNConv1d
from layers import WNConvTranspose1d
from streaming import (
    AudioBuffer,
    TensorBuffer,
)


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
            ssl_dim: int = 1024,
            d_out: int = 1,
            add_residual_connections: bool = False,
            residual_num_channels: Optional[list[int]] = None,
    ):
        super().__init__()
        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        self.ssl_upsampling_lin = nn.Linear(ssl_dim, input_channel)
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
        x = self.ssl_upsampling_lin(x)
        x = x.transpose(1, 2)
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


class EncoderWrapper(nn.Module):
    def __init__(self, ssl_src="torchaudio", layer=None):
        super().__init__()
        if ssl_src == "torchaudio":
            self.encoder = WAVLM_LARGE.get_model()
        elif ssl_src == "microsoft":
            self.encoder = self._create_microsoft_wavlm_encoder
        
        self.ssl_src = ssl_src
        self.layer = layer
        
    def _create_microsoft_wavlm_encoder(self, model_type="wavlm-base+"):
        # load the pre-trained checkpoints
        checkpoint_path = '/eph/nvme0/xkleme15/neural_codec_enhancement/pretrained_models/WavLM-Large.pt'
        checkpoint = torch.load(checkpoint_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint['model'])
        return model

    def encode(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """
            Summary:
                Encode the audio using the encoder
            :param audio: A torch tensor with the Audio
            :type audio: torch.Tensor
            :param layer: Which layer to use as features. If None, then the last
                          layer is used.
            :type layer: Optional[int] 
            :return feats: The encoded audio
            :rtype: torch.Tensor
        """
        if self.ssl_src == "torchaudio":
            feats = self.encoder.extract_features(audio, num_layers=self.layer)
            # Return the last layer
            return feats[0][-1]
        elif self.ssl_src == "microsoft":
            raise NotImplementedError

    def encode_set(self, cuts: CutSet) -> torch.Tensor:
        """
            Summary:
                Loop through audio cuts and extract the features using the encoder.
                
            :param cuts: the Lhotse cuts (CutSet) of audio files with features
                         to extract using the encoder.
            :type cuts: CutSet   
            :return: A torch tensor with the extracted features
            :rtype: torch.Tensor
        """
        device = next(self.encoder.parameters()).device
        feats = []
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16)
        ):
           # Loop throught the cuts, extract features and store them in a list
           for c in tqdm(
                cuts,
                "Collecting enrollment ..."
            ):
                audio = torch.tensor(c.load_audio()).to(device)
                encoded = self.encode(audio)
                feats.append(encoded.view(-1, encoded.size(-1)))
        return torch.cat(feats, dim=0)

    def enroll(
        self,
        cuts: CutSet,
        num_coarse_clusters: int = 1024,
        groups: int = 8,
        bits_per_code: int = 8,
    ) -> faiss.IndexIVFPQ:
        """
            Summary:
                Enroll speakers by creating an index over embeddings of their
                speech, i.e., something produced by self.encode_set
            :param cuts: The enrollment audio
            :type cuts: lhotse.CutSet
            :param num_coarse_clusters: the number of kmeans clusters used in
                                        in the FAISS index
            :type num_coarse_cluster: int
            :param groups: the number of groups to use in the product
                           quantization in the FAISS index
            :type groups: int
            :param bits_per_code: rather than specifying the number of clusters
                                  for each group in the product quantization,
                                  specify the total number of possible centroids
                                  as 2**bits_per_code
            :return: the index
            :rtype: faiss.IndexIVFPQ
        """
        # First get the embeddings to index
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16)
        ):
            data = self.encode_set(cuts)
        
            dim = data.size(-1)
            # Create the index structure
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFPQ(
                quantizer, dim, num_coarse_clusters, groups, bits_per_code
            )

            # Add the data to the index
            index.train(data.cpu())
            index.add(data.cpu())
            return index
    
    def enroll_one_speaker(
        self,
        cuts: CutSet,
        index: faiss.IndexIVFPQ,
    ) -> None:
        """
            Summary:
                Add a speaker to an existing index. The index won't be trained
                for a new speaker, which could impact the knn algorith. For best
                performance, you likely need to recreate the index using all 
                speakers.
            :param cuts: the enrollment audio
            :type cuts: lhotse.CutSet
            :param index: the existing index which we will update
            :type index: faiss.IndexIVFPQ
        """
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16)
        ):
            data = self.encode_set(cuts)
            index.add(data)


class LatentReconstructModel(nn.Module):
    def __init__(
        self,
        encoder,
        enhancer,
        decoder,
    ):
        """
            :param enhancer: the componenet resonsible for enhancing, i.e.,
                            reconstructing, the incoming audio using reference
                            latent embeddings.
            :param decoder: the decoder to use for converting from the latent
                            embedding sequences to audio
        """
        super().__init__()
        # Just assume the encoder is WAVLM for now
        self.encoder = encoder
        self.decoder = decoder
        self.enhancer = enhancer
        self.dim = self._get_output_dim()
   
    def _get_output_dim(self):
        x = torch.rand(1, 400)
        return self.encoder.encode(x).size(-1)
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder.encode(x)
        
        # Latent Enhance
        features = self.enhancer(features) 
        features = features.mean(dim=0, keepdim=True)
        
        # Decode
        audio_out = self.decoder(features)
        return audio_out.squeeze().to(torch.float32).view(1, -1).cpu()
     
    def latent_knn_generator(
        self
    ) -> Generator[torch.Tensor, np.ndarray, None]:
        """
            Summary:
                Matches the latent encodings to the enrollment encodings in
                a streaming fashion.
            :yield: a new chunk of latent embeddings
            :rtype: Generator[torch.Tensor, np.ndarray, None]
        """
        # Get the device of the encoder
        device = next(self.encoder.encoder.parameters()).device
        chunk = None
        while True:
            chunk = yield chunk
            if chunk is None:
                break

            # Extract the features:
            # Convert to a tensor and then encode the audio stream
            audio_chunk = torch.tensor(chunk, dtype=torch.float32).to(device)
            if len(audio_chunk.size()) == 1:
                audio_chunk = audio_chunk.unsqueeze(0)
            features = self.encoder.encode(audio_chunk)
            chunk = self.enhancer(features)
    
    def speech_from_latent_generator(
        self
    ) -> Generator[torch.Tensor, torch.Tensor, None]:
        """
            Summary:
                Converts a chunk of latent embeddings into audio
            :yield: a new chunk of audio
            :rtype: Generator[torch.Tensor, torch.Tensor, None]
        """
        # Get the device of the decoder
        device = next(self.decoder.parameters()).device
        chunk = None
        while True:
            chunk = yield chunk
            if chunk is None:
                break
            chunk = self.decoder(chunk)

    def speech_from_speech_generator(
        self,
        internal_buffer_size: int = 2,
    ) -> Generator[torch.Tensor, None, None]:
        """
            Summary:
                Runs the whole pipeline of taking a stream of audio into the
                model, encoding it, enchancing it, and decoding it back again.
            :param internal_buffer_size: the size of the buffer storing the
                                         encoded audio.
            :type internal_buffer_size: int 
            
            :yield: A chunk of speech
            :rtype: Generator[torch.Tensor, None, None]
        """
        # Initialize Generators
        latent_knn_gen = self.latent_knn_generator()
        next(latent_knn_gen)
        vocoder_gen = self.speech_from_latent_generator()
        next(vocoder_gen)
       
        chunk = None
        
        # Run the generator
        while True:
            chunk = yield chunk
            if chunk is None:
                break
            latent_chunk = latent_knn_gen.send(chunk)
            latent_chunk = latent_chunk.mean(dim=0, keepdim=True)
            chunk = vocoder_gen.send(latent_chunk)
    
    def stream_to_output_stream(
        self,
        stream: Generator[np.ndarray, None, None],
    ) -> Generator[torch.Tensor, None, None]:
        """
            Summary:
                Opens a file and performs streaming processing of the file
            
            :param stream: the audio stream
            :type stream: Generator[np.ndarray, None, None]
            
            :yield: a chunk of processed output speech
        """ 
        # Initialize the speech_from_speech generator
        speech_from_speech_gen = self.speech_from_speech_generator()
        next(speech_from_speech_gen)

        for d in stream:
            yield speech_from_speech_gen.send(d)

    def stream_to_tensor(
        self,
        stream: Generator[np.ndarray, None, None],
        chunk_size: int,
    ) -> torch.Tensor:
        """
            Summary:
                Just like stream_file_to_output_stream, but instead sends the
                audio to an audio tensor.
            
            :param stream: the audio stream
            :type stream: Generator[np.ndarray, None, None]
            :param chunk_size: the size of the chunk to extract
            :type chunk_size: int
            :return: the audio
            :rtype: torch.Tensor
        """
        output_chunks = []
        for d in self.stream_to_output_stream(stream):
            output_chunks.append(d.squeeze()[-chunk_size:])

        out_tensor = torch.cat(
            output_chunks,
            dim=0,
        ).to(torch.float32).view(1, -1).cpu()
        return out_tensor
             
    def smooth_buffer(self, b: torch.Tensor, offset: int) -> torch.Tensor:
        """
            Summary:
                This function takes a buffer and an offset and decides how to
                merge chunks before passing them off to the next stage of
                processing.

            :param b: the buffer to smooth buff_size x seq_len x dim
            :type b: torch.Tensor
            :param offset: the offset in frames of one sequence in the
                           buffer compared to the next
            :type offset: int
            :return: torch.Tensor with smoothed output from buffer
        """
        window_length = b.size(1)
        window = torch.hann_window(window_length).to(b.device)
        window = window.view(1, 1, window_length)
        b_permuted = b.permute(0, 2, 1)
        output = F.conv1d(b_permuted, window, padding=window_length//2)
        output = output.permute(0, 2, 1)
        return output


class NNEnhancer(nn.Module):
    def __init__(
        self,
        index: faiss.IndexIVFPQ,
        topk: int = 4,
    ): 
        super().__init__()
        self.index = index
        self.topk = topk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        _, _, R = self.index.search_and_reconstruct(
            x.view(-1, x.size(-1)).detach().cpu().numpy(), self.topk
        )
        return torch.tensor(R).transpose(0, 1).to(device)
