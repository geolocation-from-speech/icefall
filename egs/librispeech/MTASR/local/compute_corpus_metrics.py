import argparse
from lhotse import load_manifest_lazy
from lhotse.utils import compute_num_frames
import numpy as np


def compute_avg_speaker_density_per_example(start_frames_list, num_frames_list):
    """
    Args:
        start_frames_list: List of lists of segment start frames per example
        num_frames_list: List of lists of segment durations per example

    Returns:
        List of average speaker densities, one per example
    """
    avg_densities = []
    avg_overlaps = []
    avg_duration = 0
    for start_frames, num_frames in zip(start_frames_list, num_frames_list):
        assert len(start_frames) == len(num_frames)
        if len(start_frames) == 0:
            avg_densities.append(0.0)
            continue

        end_frame = max(s + n for s, n in zip(start_frames, num_frames))
        avg_duration += end_frame
        density = np.zeros(end_frame, dtype=np.int32)

        for s, n in zip(start_frames, num_frames):
            density[s:s+n] += 1

        avg_density = density.mean()
        avg_densities.append(avg_density)
        avg_overlaps.append(np.sum((density > 1)) / len(density))

    print(f"Avg dur: {avg_duration / len(start_frames_list)}")
    return avg_densities, avg_overlaps


def main(args):
    cuts = load_manifest_lazy(args.cuts)
    
    # Get the start frames and the num_frames
    starts = []
    durations = []
    audio_durs = 0
    speakers = []
    for c in cuts:
        c_starts = []
        c_durations = []
        audio_durs += c.duration
        speakers_ = set()
        for s in c.supervisions:
            speakers_.add(s.speaker)
            vals = {}
            for name, val in [("start", s.start), ("duration", s.duration)]:
                vals[name] = compute_num_frames(
                    val,
                    frame_shift=args.frame_shift,
                    sampling_rate=args.sampling_rate
                )
            c_starts.append(vals["start"])
            c_durations.append(vals["duration"])
        speakers.append(len(speakers_))
        starts.append(c_starts)
        durations.append(c_durations)
    
    densities, overlaps = compute_avg_speaker_density_per_example(starts, durations)
    print(f"Avg density: {np.array(densities).mean()}")
    print(f"Avg overlap %: {100*np.array(overlaps).mean()}")
    print(f"Avg # segs / group: {sum([len(s) for s in starts])/len(starts)}")
    print(f"Avg # spks  group: {sum(speakers) / len(starts)}")
    print(f"Avg dur {audio_durs/len(cuts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cuts")
    parser.add_argument("--frame-shift", type=float, default=0.01)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    args = parser.parse_args()
    main(args)

