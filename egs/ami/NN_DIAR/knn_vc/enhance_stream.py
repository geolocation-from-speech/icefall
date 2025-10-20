import argparse
import re
import numpy as np
from model import Decoder, LatentReconstructModel, NNEnhancer, EncoderWrapper
import torchaudio
from pathlib import Path
import faiss
import torch
from streaming import stream_audio_from_file, AudioBuffer
import time
import os
import soundfile as sf
import torch.nn.functional as F

from icefall.utils import str2bool


def make_decoder(args):
    mdl = torch.load(args.decoder, map_location="cpu")
    metadata = mdl['metadata']['kwargs']
    dec = Decoder(
        metadata['latent_dim'],
        metadata['decoder_dim'],
        metadata['decoder_rates']
    )
    state_dict_subset = {
        re.sub('decoder.','', k): v 
        for k, v in mdl['state_dict'].items() if 'decoder.' in k
    }
    state_dict_subset.update(
        {
            'ssl_upsampling_lin.weight': mdl['state_dict']['ssl_upsampling_lin.weight'],
            'ssl_upsampling_lin.bias': mdl['state_dict']['ssl_upsampling_lin.bias'],
        }
    )
    ret_val = dec.load_state_dict(state_dict_subset)
    return dec


def make_enhancer(args):
    enroll_dir = Path(args.enroll_dir)
    index = faiss.read_index(str(enroll_dir / f"{args.index_name}.faiss"))
    enhancer = NNEnhancer(index, topk=args.enhancer_topk)
    return enhancer


def make_encoder(args):
    encoder = EncoderWrapper(
        ssl_src=args.ssl_src,
        layer=args.wavlm_layer,
    ) 
    return encoder


def make_model(args):
    encoder = make_encoder(args)
    decoder = make_decoder(args)
    enhancer = make_enhancer(args)
    model = LatentReconstructModel(encoder, enhancer, decoder)
    return model


def main(args): 
    chunk_size = 2**(int(np.log2(args.sampling_rate * args.latency))) 
    buffer_size = 2**(int(np.log2(args.sampling_rate * args.buffer_duration))) 
    time_per_chunk = chunk_size / args.sampling_rate 
    
    # Then create the stream from the file mimicking and input audio stream
    stream = stream_audio_from_file(
        args.filename,
        max_chunks = args.max_chunks,
        sampling_rate = args.sampling_rate,
        chunk_size = chunk_size,
        buffer_size = buffer_size,
    )

    model = make_model(args)
    if torch.cuda.is_available():
        model.to('cuda')
        device = 'cuda'

    output_stream = model.stream_to_output_stream(stream)
    latency = []
    rtf = []
    Path(args.ofilename).parent.mkdir(parents=True, exist_ok=True)
    window = np.hanning(2*chunk_size)
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.float16)
    ):
        with sf.SoundFile(
            str(Path(args.ofilename).with_suffix(".wav")),
            mode='w',
            samplerate=args.sampling_rate,
            channels=1,
            subtype='PCM_16',
        ) as f:
            # Need to store the overlap
            prev_overlap = None 
            start = time.time()
            for d_ in output_stream:
                if prev_overlap is None:
                    prev_overlap = F.pad(d_, (2*chunk_size - d_.size(-1), 0))
                d_ = F.pad(d_, (2*chunk_size - d_.size(-1), 0))
                end = time.time()
                latency.append(end - start + time_per_chunk)
                rtf.append((end - start) / time_per_chunk)
                d_.squeeze()[-chunk_size:]
                prev_overlap[0:d_.squeeze().to(torch.float32).cpu().numpy()
                #out_chunk = d_.squeeze()[-chunk_size:].to(
                #    torch.float32
                #).cpu().numpy()
                f.write(out_chunk)
                start = time.time()
              
    print(f"latency mean: {np.mean(latency)} std: {np.std(latency)}")
    print(f"rtf mean: {np.mean(rtf)} std: {np.std(rtf)}")
  
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="the file to stream")
    parser.add_argument("ofilename", type=str, help="the saved converted file")
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="The maximum number of chunks to stream"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=75,
        help="The maximum duration in seconds of audio to convert",
    )
    parser.add_argument("--latency", type=float, default=0.2,
        help="The desired latency of the system minus what is incurred by RTF"
    )
    parser.add_argument("--buffer-duration", type=float, default=0.4,
        help="Size of the audio buffer to pass to the model."
    )
    parser.add_argument("--decoder", type=str,
        help="path to a pretrained decoder"
    )
    parser.add_argument("--ssl-src", type=str, default="torchaudio",
        help="which WavLM model to use"
    )
    parser.add_argument(
        "--enroll-dir",
        type=str,
        default="enrollment",
        help="The directory in which to save the enrollment index",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="speakers",
    )
    parser.add_argument(
        "--wavlm-layer",
        type=int,
        default=None,
        help="The layer at which we extract embeddings for matching"
    )
    parser.add_argument(
        "--enhancer-topk",
        type=int,
        default=4,
        help="How many embeddings to return for enhancement regression"
    )
    parser.add_argument(
        "--output-stats", type=str, default=None,
    )
    
    args = parser.parse_args() 
    main(args)
