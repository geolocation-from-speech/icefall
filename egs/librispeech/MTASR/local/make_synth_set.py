from lhotse.dataset.sampling.cut_splice3 import CutSpliceIterable
from lhotse import load_manifest_lazy, CutSet
import argparse
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("outpath")
    parser.add_argument("duration", type=float, help="duration in hrs")
    parser.add_argument("--num-spks", type=int, default=2)
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument("--overlap", type=float, default=1.0)
    parser.add_argument("--max-num-overlaps", type=int, default=3)
    parser.add_argument("--max-duration", type=int, default=30)
    parser.add_argument("--num-splices", type=int, default=2)

    args = parser.parse_args()

    #outpath = "data/manifests2/cuts_librispeech_dev_synth2.jsonl.gz" 
    outpath = args.outpath

    speakers = list(Path("data/manifests2/librispeech_dev_clean_speakers").rglob("*.jsonl.gz"))
    num_spks = len(speakers)
    cut_info = []
    for s in speakers:
        cut_info.append((s.stem.split("_")[-1].split(".")[0], 1/num_spks, s))  
    cutsets, weights, names = [], [], []
    for n, w, p in cut_info: 
        cutsets.append(load_manifest_lazy(p))
        weights.append(w)
        names.append(n)

    cs_iter = CutSpliceIterable(
        cutsets,
        cutset_weights=weights,
        cutset_prefixes=names,
        max_duration=args.max_duration,
        final_max_duration=args.max_duration,
        max_splices=args.num_splices,
        min_splices=args.num_splices,
        max_snr=[0 for i in range(num_spks)],
        final_min_splices=args.num_splices,
        final_max_splices=args.num_splices,
        max_unique=4*args.num_spks,
        normalize_loudness=False,
        sampling_rate=16000,
        self_overlap=False,
        min_overlap=args.min_overlap,
        overlap=args.overlap,
        max_num_overlaps=args.max_num_overlaps,
    )
    
    # Just do two hours of data
    total_duration = 0
    cuts = []
    with tqdm(total=args.duration*3600) as pbar:
        for i, c in enumerate(cs_iter):
            if total_duration > args.duration*3600:
                break
            num_spks = len(set([s.speaker for s in c.supervisions]))
            num_splices = len(c.supervisions)
            if num_spks != args.num_spks or num_splices != args.num_splices:
                continue
            total_duration += c.duration
            cuts.append(c)
            if i % 10 == 0:
                pbar.n = round(total_duration, 1)
                pbar.refresh() 
        pbar.n = round(total_duration, 1)
        pbar.refresh() 

    CutSet(cuts).to_file(outpath)
