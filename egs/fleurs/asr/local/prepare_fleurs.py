#!/usr/bin/env python3
import argparse
from lhotse import load_manifest_lazy, CutSet
from pathlib import Path
from tqdm import tqdm
import json


def main(args):
    recos = load_manifest_lazy(args.recos)
    sups = load_manifest_lazy(args.sups)
    cuts = CutSet.from_manifests(recordings=recos, supervisions=sups)
    cuts.to_file(args.cuts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("recos")
    parser.add_argument("sups")
    parser.add_argument("cuts")
    args = parser.parse_args()
    main(args)
