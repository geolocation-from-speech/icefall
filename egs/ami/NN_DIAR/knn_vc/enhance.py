import argparse
import re
import numpy as np
from model import Decoder, LatentReconstructModel, NNEnhancer, EncoderWrapper
import torchaudio
from pathlib import Path
import faiss
import torch
import soundfile as sf
from torchaudio.transforms import Resample

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
    enhancer = NNEnhancer(index)
    return enhancer


def make_encoder(args):
    encoder = EncoderWrapper(ssl_src=args.ssl_src) 
    return encoder


def make_model(args):
    encoder = make_encoder(args)
    decoder = make_decoder(args)
    enhancer = make_enhancer(args)
    model = LatentReconstructModel(encoder, enhancer, decoder)
    return model


def main(args):
    model = make_model(args)
    if torch.cuda.is_available():
        model.to('cuda')
        device = 'cuda'

    with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16)
        ):
        # Load the audio file
        audio, sr = sf.read(args.filename)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)

        device = next(model.parameters()).device 
        audio = audio[59*16000:70*16000]
        audio = torch.tensor(audio, dtype=torch.float32).to(device)
        resampler = Resample(orig_freq=sr, new_freq=args.sampling_rate)
        resampled_audio = resampler(audio)
       
        output = model(resampled_audio.unsqueeze(0)) 

        Path(args.ofilename).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            Path(args.ofilename).with_suffix(".wav"),
            output,
            args.sampling_rate,
            format="wav"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="the file to stream")
    parser.add_argument("ofilename", type=str, help="the saved converted file")
    parser.add_argument("--sampling-rate", type=int, default=16000)
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
    args = parser.parse_args() 
    main(args)
