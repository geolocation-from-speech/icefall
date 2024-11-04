import torch
import numpy as np
from pathlib import Path
import argparse
from lhotse import load_manifest_lazy
from lhotse import CutSet
import math
from tqdm import tqdm


def angular_distance(Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    '''
        angular distance points are (lat, long). Rows are points, cols are
        lat, lon. Values are assumed to be in radian.

        :param X: The tensor of hypothesis locations (lat,lon coordinates)
        :param Y: The tensor of target locations (lat,lon coordinates)
        :return: The angular distance between each element in X and the
            corresponding element in Y.
    '''
    h = (
        torch.sin(X[:, 0])*torch.sin(Y[:, 0]) + 
        torch.cos(X[:, 0])*torch.cos(Y[:, 0])*torch.cos(X[:, 1] - Y[:, 1])
    )
    return torch.acos(h.clamp(-1, 1))


def main(args):
    resultsdir = Path(args.results)
    preds = resultsdir.rglob(f"preds*{args.suffix}.npy")
    tgts = resultsdir.rglob(f"tgts*{args.suffix}.npy")
    ids = resultsdir.rglob(f"ids*{args.suffix}.npy")
    cuts = load_manifest_lazy(args.cuts)

    keep_ids = set()
    for p, t, i in tqdm(zip(preds, tgts, ids), "Finding bad data"):
        p_ = np.load(p)
        t_ = np.load(t)
        i_ = np.load(i) 
        p_ = torch.from_numpy(p_ * (math.pi / 180))
        t_ = torch.from_numpy(t_ *  (math.pi / 180))

        dists = 6378.1 * angular_distance(t_, p_)
        keep_ids.update([cut_id for cut_id, d in zip(i_, dists) if d < args.threshold])

    
    with CutSet.open_writer(args.out) as cut_writer:
        for c in tqdm(cuts.filter(lambda c: c.id in keep_ids), "Writing cleaned cuts to file"):
            cut_writer.write(c) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuts")
    parser.add_argument("--results")
    parser.add_argument("--suffix")
    parser.add_argument("--out")
    parser.add_argument("--threshold", type=int, default=5000)
    args = parser.parse_args()
    main(args)
