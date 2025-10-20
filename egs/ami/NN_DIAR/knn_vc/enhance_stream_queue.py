from pathlib import Path
import faiss
import torch
import multiprocessing as mp
from streaming import stream_audio_from_file
import time
import os

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


#def cpu_generator(args, q):
#    # ---- Create the audio stream from a file ------
#    # First get the chunk_size from the desired latency
#    chunk_size = 2**(int(np.log2(args.sampling_rate * args.latency))) 
#    buffer_size = 2**(int(np.log2(args.sampling_rate * args.buffer_duration))) 
#    time_per_chunk = chunk_size / args.sampling_rate 
#    
#    # Then create the stream from the file mimicking and input audio stream
#    stream = stream_audio_from_file(
#        args.filename,
#        max_chunks = args.max_chunks,
#        sampling_rate = args.sampling_rate,
#        chunk_size = chunk_size,
#        buffer_size = buffer_size,
#    )
#    for d in stream:
#        q.put(d)
#        if time_per_chunk > 0:
#            time.sleep(time_per_chunk)
#    q.put(None)
#       
#        
#def gpu_consumer(args, input_q, output_q, gpu_id): 
#    print(os.environ["CUDA_VISIBLE_DEVICES"])
#    print(torch.cuda.is_available())
#    print(torch.cuda.device_count())
#    print(torch.cuda.current_device())
#    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#    torch.cuda.set_device(0)
#    torch.set_num_threads(1)
#    torch.set_num_interop_threads(1)
#    device = torch.cuda.current_device() 
#    model = make_model(args)
#    #if torch.cuda.is_available():
#    #    model.to('cuda')
#    #    device = 'cuda'
#
#    generator = model.speech_from_speech_generator()
#    next(generator)
#
#    with (
#        torch.inference_mode(),
#        torch.autocast(device_type="cuda", dtype=torch.float16)
#    ):
#        while True:
#            chunk = input_q.get()
#            if chunk is None:
#                break
#            #output = model(chunk)
#            output = generator.send(chunk)
#            output_q.put(output)
        

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

    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.float16)
    ):
        output = model.stream_to_tensor(stream, chunk_size)
    #generator = model.speech_from_speech_generator()
    #next(generator)


#    input_q = mp.SimpleQueue()
#    output_q = mp.Queue()
#    
#    gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]) 
#    # Start GPU worker
#    gpu_process = mp.Process(
#        target=gpu_consumer,
#        args=(args, input_q, output_q, gpu_id)
#    )
#
#    gpu_process.start()
#
#    # Start CPU producer
#    cpu_process = mp.Process(
#        target=cpu_generator,
#        args=(args, input_q)
#    )
#    cpu_process.start()
#
#    output_chunks = [] 
#    chunk_size = 2**(int(np.log2(args.sampling_rate * args.latency))) 
#    # Read results asynchronously
#    while cpu_process.is_alive() or not output_q.empty():
#        try:
#            result = output_q.get()  # Get result if available
#            output_chunks.append(result.squeeze()[-chunk_size:].cpu())
#            print(len(output_chunks))
#        except mp.queues.Empty:
#            pass  # No result yet
#
#    # Cleanup
#    gpu_process.join()
#    cpu_process.join()
#    input_queue.close()
#    output_queue.close() 
#    import pdb; pdb.set_trace()  
    #output = torch.cat(output_chunks, dim=0).to(torch.float32).view(1, -1)
    #output = model.stream_to_tensor(stream, chunk_size)
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
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=156,
        help="The maximum number of chunks to stream"
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
        "--emulate-time-passage", type=str2bool, default=False,
            help="Artificially stops the stream for "
            "sampling_rate / chunk_width seconds to emulate the passage of "
            "time in a real audio stream."
    )
    args = parser.parse_args() 
    mp.set_start_method("spawn", force=True)
    main(args)
