#!/usr/bin/env python
from lhotse import CutSet
import argparse
from pathlib import Path
from tqdm import tqdm
import unicodedata
import re


def main(args):
    cuts = CutSet.from_jsonl_lazy(args.cuts)
    ofile = Path(args.ofile)
    ofile.parent.mkdir(mode=511, exist_ok=True, parents=True)
    sups = set()
    with open(args.ofile, 'w', encoding='utf-8') as f:
        for c in tqdm(cuts):
            for s in c.supervisions:
                #text = unicodedata.normalize("NFKC", s.text)
                #text = re.sub(r"  +", " ", text)
                #text = re.sub(r"^ ", "", text)
                print(s.text, file=f)
                sups.add(s.id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cuts", help="Path to training cut manifests")
    parser.add_argument("ofile", help="path to the output transcripts")
    args = parser.parse_args()
    main(args)
