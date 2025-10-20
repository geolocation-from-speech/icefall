import numpy as np
import argparse
from pathlib import Path


def main(args):
    embeds1 = np.load(args.embeds1)
    embeds2 = np.load(args.embeds2)
    tgts1 = np.load(args.tgts1)
    tgts2 = np.load(args.tgts2)

    dirname = Path(args.embeds1).parent
    suffix1 = Path(args.embeds1).stem.split("_", 1)[1]
    suffix2 = Path(args.embeds2).stem.split("_", 1)[1]
      
    langs = sorted(set(tgts1).union(set(tgts2)))

    ids1 = np.load(f"{dirname}/ids_{suffix1}.npy")
    ids2 = np.load(f"{dirname}/ids_{suffix2}.npy")

    embeds = np.concatenate((embeds1, embeds2))
    tgts   = np.concatenate((tgts1, tgts2))
    ids    = np.concatenate((ids1, ids2))

    assert args.outname != str(Path(args.embeds1).stem)
    assert args.outname != str(Path(args.embeds2).stem)

    np.save(dirname / f"embeds_{args.outname}.npy", embeds)
    np.save(dirname / f"tgts_{args.outname}.npy", tgts)
    np.save(dirname / f"ids_{args.outname}.npy", ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Combine numpy embeddings across splits"
    )
    parser.add_argument("--embeds1", type=str, required=True)
    parser.add_argument("--tgts1", type=str, required=True)
    
    parser.add_argument("--embeds2", type=str, required=True)
    parser.add_argument("--tgts2", type=str, required=True)
    
    parser.add_argument("--outname", type=str, required=True)
    args = parser.parse_args()
    main(args)
