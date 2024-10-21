#!/usr/bin/env python3
import argparse
from lhotse import Recording, SupervisionSegment, SupervisionSet, RecordingSet, CutSet
from pathlib import Path
from tqdm import tqdm
import json


def main(args):
    recos = []
    sups = []
    odir = Path(args.odir)
    odir.mkdir(mode=511, parents=True, exist_ok=True)
    with open("fleurs_geolocations.json", "r") as f:
        languages = json.load(f)
    
    for d in args.audio_dirs:
        d = Path(d)
        langid = d.parent.stem
        files = d.rglob("*.wav")
        for f in tqdm(files, f"Processing file from {langid}"):
            reco = Recording.from_file(f)
            recos.append(reco)
            sups.append(
                SupervisionSegment(
                    id=reco.id,
                    recording_id=reco.id,
                    start=0,
                    duration=reco.duration,
                    language=langid,
                    custom={
                        'lat': languages[langid]['lat'],
                        'lon': languages[langid]['lon'],
                    }
                )
            )
    
    recordings = RecordingSet.from_recordings(recos)
    supervisions = SupervisionSet.from_segments(sups)
    cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
    cuts.to_file(str(odir / f"cuts.jsonl.gz"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("odir", type=str)
    parser.add_argument("audio_dirs", nargs="+", type=str)
    args = parser.parse_args()
    main(args)
