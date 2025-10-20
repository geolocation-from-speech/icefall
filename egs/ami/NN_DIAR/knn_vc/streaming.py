from pathlib import Path
import io
from tqdm import tqdm
import numpy as np
from typing import Optional, Generator
import soundfile as sf
from torchaudio.transforms import Resample
import torch


class AudioBuffer:
    """
        This class implements a circular buffer
    """
    def __init__(
        self,
        buffer_size: int = 32000,
    ):
        """
           :param buffer_size: the size of the audio buffer (samples)
           :type buffer_size: int
        """
        self.buffer_size = buffer_size 
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        
        self.curr_idx = 0
        self.is_full = False

    def enqueue(self, chunk: np.ndarray):
        """
        Summary:
            Enqueue the next chunk.
            
            :param chunk: the chunk of audio (np.ndarray)
            :type chunk: np.ndarray 
        """
        start = self.curr_idx
        end = start + len(chunk)

        # Implements a circular buffer
        if end < self.buffer_size:
            self.audio_buffer[start:end] = chunk
        else:
            self.is_full = True
            what_fits = self.buffer_size - start
            self.audio_buffer[start:] = chunk[:what_fits]
            self.audio_buffer[:end % self.buffer_size] = chunk[what_fits:]
        
        self.curr_idx = end % self.buffer_size

    def get_buffer(self) -> np.ndarray:
        """
        Summary:
            Get the current buffer content in the correct order.
            
            :return: The latest buffered audio data in the correct order
            :rtype: np.ndarray
        """
        if not self.is_full:
            # Return only filled portion
            return self.audio_buffer[:self.curr_idx]
        
        return np.roll(self.audio_buffer, -self.curr_idx)


class TensorBuffer:
    """
        This class implements a circular buffer
    """
    def __init__(
        self,
        buffer_size: int = 100,
        seq_len: int = 3, 
        dim: int = 1024,
    ):
        """
           :param buffer_duration: the duration of the audio buffer (s)
           :type buffer_duration: float
           :param sampling_rate: the sampling rate of the audio we are ingesting
           :type sampling_rate: int
        """
        self.buffer_size = buffer_size 
        self.seq_len = seq_len
        self.tensor_buffer = torch.empty(buffer_size, seq_len, dim) 
        
        self.curr_idx = 0
        self.is_full = False

    def enqueue(self, chunk: torch.Tensor):
        """
        Summary:
            Enqueue the next chunk.
            
            :param chunk: the chunk of audio (np.ndarray)
            :type chunk: torch.Tensor 
        """
        start = self.curr_idx
        end = start + len(chunk)

        # Implements a circular buffer
        if end <= self.buffer_size:
            self.tensor_buffer[start:end, :, :] = chunk
        else:
            self.is_full = True
            what_fits = self.buffer_size - start
            self.tensor_buffer[start:, :, :] = chunk[:what_fits]
            self.audio_buffer[:end % self.buffer_size, :, :] = chunk[what_fits:]
        
        self.curr_idx = end % self.buffer_size

    def get_buffer(self) -> np.ndarray:
        """
        Summary:
            Get the current buffer content in the correct order.
            
            :return: The latest buffered audio data in the correct order
            :rtype: np.ndarray
        """
        if not self.is_full:
            # Return only filled portion
            return self.tensor_buffer[0, :self.curr_idx, :]
        
        return self.tensor_buffer.roll(-self.curr_idx, dims=1)


def stream_audio_from_file(
    filepath: str,
    max_chunks: Optional[int] = 56250,
    max_duration: Optional[float] = 76, 
    sampling_rate: Optional[int] = 16000,
    chunk_size: Optional[int] = 1024,
    buffer_size: Optional[int] = 8192,
    latency: float = 0.2,
    window: str = "hann",
    dtype: str = 'int16',
) -> Generator[np.ndarray, None, None]:
    """
        Summary:
            Open up a file for streaming and read from the stream
            chunk-by-chunk until the stream is exhausted.
            
            :param filepath: the path to the file to convert to an audio stream
            :type filepath: str
            
            :param max_chunks: the maximum number of chunks to ingest
            :type max_chunks: int

            :param max_duration: the maximum duration of the audio to stream
            :type max_duration: float (s)
            
            :param sampling_rate: the desired sampling rate of the speech
            :type sampling rate: int

            :param latency: the desired latency, i.e., in time for the system.
                            it will help determine things like the chunk_size
            :type latency: float

            :param dtype: the input byte type
            :type dtype: str

            :param window: the window used for overlap add
            :type window: str
    """
    # If the sampling rate is None, then we will just use whatever the
    # default sampling rate is of the loaded file. We will set the other
    # parameters to default values if they are also specified to be None,
    # based on the sampling rate.
    if sampling_rate is None:
        sampling_rate = 16000
    
    # If the chunk_size is None, then we want to set it to correspond to
    # about 200 ms latency. We stick to powers of 2 and round down.
    if chunk_size is None:
        chunk_size = 2**(int(np.log2(sampling_rate * latency)))
     
    audio_buffer = AudioBuffer(buffer_size=buffer_size)
    # If max_chunks is None, set to correspond to about 1hr of speech.
    HR_OF_SPEECH_IN_SECONDS = 3600
    if max_chunks is None and max_duration is None:
        max_chunks = (HR_OF_SPEECH_IN_SECONDS * sampling_rate) // chunk_size
    elif max_chunks is None:
        max_chunks = (max_duration * sampling_rate) // chunk_size
    
    time_per_chunk = chunk_size / sampling_rate
    print("chunk_size: ", chunk_size)
    print("chunk_time: ", time_per_chunk)
    print("audio_time: ", time_per_chunk * max_chunks)
    print("buffer_size: ", buffer_size)

    # Load data
    data, sr = sf.read(filepath, dtype='int16')
    
    # We want a single channel in the end, so we just mix the channels.
    if len(data.shape) > 1:
        data = np.mean(data, axis=0)
   
    # Change to desired sampling rate
    if sr != sampling_rate:
        data_tensor = torch.tensor(data).float()
        
        # Resample
        resampler = Resample(
            orig_freq=sr, new_freq=target_sr
        )
        resampled_audio = resampler(audio_tensor)

    # Convert the audio file into a byte stream
    audio_buffer_full = io.BytesIO(data.tobytes())
    
    # The plus 1 is to make sure to not clip the end at all
    num_chunks = len(data) // chunk_size + 1
    filename = Path(filepath).name
    for i in tqdm(
        range(min(num_chunks, max_chunks)),
        f"Streaming {filename} ..."
    ):
        try:
            # Since each sample is an int16, we need to read 2 bytes at
            # a time
            chunk = audio_buffer_full.read(chunk_size * 2)
            if len(chunk) == 0:
                break
                
            # Dividing by 32768 converts the int16 to a float
            chunk_float = np.frombuffer(chunk, dtype=np.int16) / 32768.
            audio_buffer.enqueue(chunk_float)
            yield audio_buffer.get_buffer()
        except KeyboardInterrupt:
            return  

