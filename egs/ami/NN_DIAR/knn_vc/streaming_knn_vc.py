#!/usr/bin/env python
# coding: utf-8

import soundfile as sf
import time
import io
import numpy as np
import torch
import faiss
import torch.nn as nn
from tqdm import tqdm
from torchaudio.pipelines import WAVLM_LARGE
from lhotse import Recording, CutSet
import argparse
from pathlib import Path
from functools import partial

# Define some useful variables
SAMPLE_RATE=16000
CHUNK_SIZE=4096


# Point to a test file
test_file = "download/amicorpus/ES2011a/audio/ES2011a.Mix-Headset.wav"


def read_from_buffer(buffer, chunk_size):
    chunk = buffer.read(chunk_size)
    return chunk


def stream_audio_from_file(filename, max_chunks=56250):
    if max_chunks is None:
        max_chunks = 56250
    data, samplerate = sf.read(filename, dtype='int16')
    if len(data.shape) > 1:
        data = data[:, 0] # Take only the first channel
    
    # I only want a small part of the data for now (10s)
    data = data[0:int(SAMPLE_RATE * 10)]  
    audio_buffer = io.BytesIO(data.tobytes())
    num_chunks = len(data) // CHUNK_SIZE
    for i in tqdm(range(min(num_chunks, max_chunks))):
        try:
            chunk = read_from_buffer(audio_buffer, CHUNK_SIZE * 2)
            if len(chunk) == 0:
                break
            # This mimics the true passage of time
            #time.sleep(CHUNK_SIZE / SAMPLE_RATE)
            input_ = np.frombuffer(chunk, dtype=np.int16) / 32768
            yield input_
        except KeyboardInterrupt:
            pass
        

# This function makes the FAISS index. It implements the NN-search
def make_index(data,
    num_coarse_clusters=100,
    m=8,
    bits_per_code=8,
    dim=1024,
    num_points=1500000
):
    quantizer = faiss.IndexFlatL2(dim)  # this remains the same
    index = faiss.IndexIVFPQ(
        quantizer, dim, num_coarse_clusters, m, bits_per_code
    ) 
    
    if data is None:
        data = np.random.random((1500000, dim)).astype('float32') - 0.5
    
    index.train(data)
    index.add(data)
    return index


# This generator implements the streaming NN-reconstructed features
def streaming_nn(fn, index, test_file, nprobe=2, topk=4, max_chunks=4, buffer_size=10):
    index.nprobe = nprobe # let's just search two clusters for now.
    feats_buffer = None
    for d in tqdm(stream_audio_from_file(test_file, max_chunks=max_chunks)):
        chunk_data = torch.tensor(d, dtype=torch.float32).to('cuda')
        feats = fn(chunk_data.view(1, -1))[0][-1]
         
        # Initialize the buffer for the first chunk
        if feats_buffer is None:
            feats_buffer = torch.empty(
                (1, 0, feats.size(-1)),
                device=feats.device
            )
        feats_buffer = torch.cat([feats_buffer, feats], dim=1)

        # Implement queue
        if feats_buffer.size(1) > buffer_size:
            feats_buffer = feats_buffer[:, -buffer_size:, :]

        _, _, R = index.search_and_reconstruct(
            feats.squeeze(0).detach().cpu().numpy(), topk
        )
        yield torch.tensor(R).transpose(0, 1).to('cuda'), feats


def collect_matching_set(cuts, fn, device='cpu'):
    feats = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for c in tqdm(cuts, "Collecting matching set ..."):
            out = fn(torch.tensor(c.load_audio()).to(device))[0][-1]
            feats.append(out.view(-1, out.size(-1)))
    return torch.cat(feats, dim=0)


def enroll(args):    
    # This will become the matching set
    cuts = CutSet.from_manifests(
        recordings=[Recording.from_file(test_file)]
    ).cut_into_windows(10).to_eager()

    # Load the wavlm model
    mdl = WAVLM_LARGE.get_model() 
    
    device = 'cpu'
    # Check that cuda is available
    if torch.cuda.is_available():
        mdl.to('cuda')
        device = 'cuda'
  
    fn = partial(mdl.extract_features, num_layers=args.wavlm_layer)
    matching_set = collect_matching_set(cuts, fn, device='cuda') 
    
    # Create the index. For now we use the test audio file as dummy embeddings
    # for the real audio
    index = make_index(
        matching_set.detach().cpu().numpy(),
        num_coarse_clusters=1024
    )
    
    # Make the directory where we will store the enrollment index
    enroll_dir = Path(args.enroll_dir)
    enroll_dir.mkdir(parents=True, exist_ok=True)

    # Store the index
    faiss.write_index(index, str(enroll_dir / f"{args.index_name}.faiss")) 


def enhance(args):
    enroll_dir = Path(args.enroll_dir)

    # Load the index
    index = faiss.read_index(str(enroll_dir / f"{args.index_name}.faiss"))
   
    # Load the wavlm model
    mdl = WAVLM_LARGE.get_model() 
    
    # Check that cuda is available
    if torch.cuda.is_available():
        mdl.to('cuda')
        device = 'cuda'
         
    fn = partial(mdl.extract_features, num_layers=args.wavlm_layer)

    # Stream the audio, matching to the matching set and check the difference.
    # The difference really should be close to 0
    err = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for d_, d in streaming_nn(fn, index, test_file,
            topk=1, buffer_size=10, max_chunks=None
        ):
            err.append(((d_ - d)**2).sum() / CHUNK_SIZE)

    print(err)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--mode",
        choices=["enroll", "enhance"],
        help="choose which mode to run in."
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
    if args.mode == "enroll":
        enroll(args)
    elif args.mode == "enhance":
        enhance(args)


