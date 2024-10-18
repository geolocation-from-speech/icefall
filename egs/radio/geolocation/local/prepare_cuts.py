#!/usr/bin/env python3
import argparse
from lhotse import load_manifest_lazy, fix_manifests, CutSet
from tqdm import tqdm
from pathlib import Path


valid_stations="""m4VESYpw
SiJV_lYK
SYCyZWnI
uztnKK23
MUq5WAJ8
AtjC0lVm
hFtZdlD7
D9W8VNEe
ThwOsInM
HHikF5mO
KQF445vh
5awguDj0
AnP3NyFu
uZeYNZzC
rrGxSGIb
vwwkadDl
5TGMtep5
puYSMon7
vcZg0CDI
rI3loVta
tl357Kz5
7Iz7yw5K
Mzcoa0KE
dvh1nQq1
Fz_77-J9
BA0Oh4Qu
wAsFnJLS
xxbjL4cw
I2C6Ww3m
BzAIOYkL
SMuks16K
Rj35oQRS
Kep0jAq_
Q-cyIHST
d9zbZG28
GY2CPOWQ
CQAAS1K7
FirzXrVe
CtPgWJda
gPGeP1Ly
CP4KHede
EsGd5eWD
5OtZR0vB
vjUza7a5
9urNOQkl
pfPYAQRu
GGxK0ewe
mHrRIfhQ
RD_IF7Lp
XybUjPHm"""


valid_stations = valid_stations.split("\n")


def main(args):
    recos = load_manifest_lazy(args.recordings)
    sups = load_manifest_lazy(args.supervisions)
    recos, sups = fix_manifests(recos, sups)
    cuts = CutSet.from_manifests(recordings=recos, supervisions=sups)
    cuts = cuts.trim_to_supervisions(
        keep_overlapping=False,
        keep_all_channels=False
    )
    out = Path(args.output_cuts)
    out_stem = Path(args.output_cuts.replace(".jsonl.gz", "")).stem
    window_lengths = [int(w) for w in args.window_lengths.split(",")]
    for window_length in tqdm(window_lengths):
        if window_length > 0:
            cuts_windowed = cuts.cut_into_windows(window_length)
        else:
            cuts_windowed = cuts
       
        out_window_length = out.with_stem(
            f"{out_stem}.{window_length}.jsonl"
        ).with_suffix(".gz")
        with CutSet.open_writer(out_window_length) as cut_writer:
            for cut in tqdm(cuts_windowed):
                cut_writer.write(cut)
    
    # Do the valid cuts
    valid_cuts = cuts.filter(lambda c: c.duration >= 2)
    valid_cuts = valid_cuts.filter(
        lambda c: c.supervisions[0].custom['station'] in valid_stations
    )
    out_valid = out.with_stem(f"{out_stem}.valid.jsonl").with_suffix(".gz")
    with CutSet.open_writer(out_valid) as cut_writer:
        for cut in tqdm(valid_cuts):
            cut_writer.write(cut)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", required=True)
    parser.add_argument("--supervisions", required=True)
    parser.add_argument("--window-lengths", type=str, default="0")
    parser.add_argument("output_cuts")
    args = parser.parse_args()
    main(args)
