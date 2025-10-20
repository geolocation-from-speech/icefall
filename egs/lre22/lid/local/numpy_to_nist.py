import numpy as np
import argparse


langs = [
    "afr-afr",
    "ara-aeb",
    "ara-arq",
    "ara-ayl",
    "eng-ens",
    "eng-iaf",
    "fra-ntf",
    "nbl-nbl",
    "orm-orm",
    "tir-tir",
    "tso-tso",
    "ven-ven",
    "xho-xho",
    "zul-zul",
]


lang2int = {l: i for i, l in enumerate(langs)}


int2lang = {i: l for i, l in enumerate(langs)}


def main(args):
    scores = np.load(args.numpy_scores)
    ids = np.load(args.numpy_ids)

    with open(args.nist, "w") as f:
        print(f"segmentid\t" + "\t".join(langs), file=f)
        tab = "\t"
        for i, uttid in enumerate(sorted(ids)):
            print(f"{uttid}\t{tab.join([str(s) for s in scores[i]])}", file=f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--numpy-scores", type=str, required=True)
    parser.add_argument("--numpy-ids", type=str, required=True)
    parser.add_argument("--nist", type=str, required=True)
    args = parser.parse_args()
    main(args)
