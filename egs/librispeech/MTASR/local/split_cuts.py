#!/usr/bin/env python
from lhotse import load_manifest_lazy, RecordingSet, SupervisionSet, CutSet
import argparse
from pathlib import Path
from tqdm import tqdm


def main(args):
    if args.recos is not None and args.sups is not None:
        cuts = CutSet([])
        for r, s in zip(args.recos, args.sups):
            recos = RecordingSet.from_jsonl(r)
            sups = SupervisionSet.from_jsonl(s)
            cuts_ = CutSet.from_manifests(recordings=recos, supervisions=sups)
            if args.trim_to_supervisions:
                cuts_ = cuts_.trim_to_supervisions(
                    keep_overlapping=False, keep_all_channels=False
                )
            cuts = cuts + cuts_
    elif args.cuts is not None:
        cuts = load_manifest_lazy(args.cuts)
        if args.trim_to_supervisions:
            cuts = cuts.trim_to_supervisions(
                keep_overlapping=False, keep_all_channels=False,
            )
    else:
        raise ValueError(
            "Either both sups and recos, or just cuts must be specified"
        )   
    # We do this so that there is an even mix of file types in each split which
    # makes subsequent feature extraction much faster
    if not args.skip_shuffle:
        cuts = cuts.shuffle()
    
    odir = Path(args.odir)
    odir.mkdir(mode=511, parents=True, exist_ok=True)
    prefix = f"cuts_{args.oname}" if args.oname is not None else f"cuts"
    import pdb; pdb.set_trace()
    for cut_idx, cut_set in tqdm(enumerate(cuts.split(args.num_splits))):
        cut_set.to_file(str(odir / f"{prefix}_{cut_idx}.jsonl.gz"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recos", nargs='+', type=str)
    parser.add_argument("--sups", nargs='+', type=str)
    parser.add_argument("--cuts", type=str)
    parser.add_argument("odir")
    parser.add_argument("--num-splits", type=int, default=200)
    parser.add_argument("--trim-to-supervisions", action="store_true")
    parser.add_argument("--oname", type=str, default=None)
    parser.add_argument("--skip-shuffle", action="store_true")
    args = parser.parse_args()
    main(args)
