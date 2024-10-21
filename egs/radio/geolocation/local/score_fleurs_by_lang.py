import torch
import numpy as np
from pathlib import Path
import argparse
from itertools import groupby
from lhotse import load_manifest_lazy
import math
import json
from tqdm import tqdm


def spherical_to_cartesian(preds):
    cos_lat = torch.cos(preds[:, 0]).unsqueeze(-1)
    cos_lon = torch.cos(preds[:, 1]).unsqueeze(-1)
    sin_lat = torch.sin(preds[:, 0]).unsqueeze(-1)
    sin_lon = torch.sin(preds[:, 1]).unsqueeze(-1)
    preds_cartesian = torch.cat(
        [
            cos_lat * cos_lon,
            cos_lat * sin_lon,
            sin_lat
        ], dim=1
    )
    return preds_cartesian 


def cartesian_to_spherical(preds):
    preds = preds / preds.norm()
    preds_lon = torch.atan2(preds[:, 1], preds[:, 0]).unsqueeze(-1) 
    preds_lat = torch.asin(preds[:, 2]).unsqueeze(-1)
    preds_spherical = torch.cat((preds_lat, preds_lon), dim=1)
    return preds_spherical


def angular_distance(Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    h = (
            torch.sin(X[:, 0])*torch.sin(Y[:, 0]) +
            torch.cos(X[:, 0])*torch.cos(Y[:, 0])*torch.cos(X[:, 1] - Y[:, 1])
        )
    return torch.acos(h.clamp(-1, 1))


def main(args):
    results_dir = Path(args.results_dir)
    preds = np.load(Path(args.results_dir) / f"preds_{args.suffix}.npy")
    tgts = np.load(Path(args.results_dir) / f"tgts_{args.suffix}.npy")
    ids = np.load(Path(args.results_dir) / f"ids_{args.suffix}.npy")
    ids = dict(map(reversed, enumerate(ids)))

    # Upper bound on distance in km
    dists = 6378.1 * angular_distance(
        torch.from_numpy(tgts * (math.pi / 180)),
        torch.from_numpy(preds * (math.pi / 180)),
    )
    cuts = load_manifest_lazy(args.cuts)
    languages = {}
    for c in tqdm(cuts):
        lang = c.supervisions[0].language
        if lang not in languages:
            languages[lang] = {'dists': [], 'preds': []}
        idx = ids[c.id]
        languages[lang]['dists'].append(dists[idx])
        languages[lang]['preds'].append(preds[idx] * (math.pi / 180)) 

    results = {l: {} for l in languages}
    for l in languages:
        results[l]['avg_dist'] = sum(languages[l]['dists']).item() / len(languages[l]['dists'])
        preds_l = torch.from_numpy(np.array(languages[l]['preds']))
        preds_l_cartesian = spherical_to_cartesian(preds_l)
        total_l_pred = preds_l_cartesian.sum(dim=0)
        total_l_pred /= total_l_pred.norm()
        total_l_pred = cartesian_to_spherical(total_l_pred.unsqueeze(0))
        tgts_l = total_l_pred * torch.ones(len(languages[l]['preds']), 2)
        dists_l = 6378.1 * angular_distance(tgts_l, preds_l) 
        results[l]['avg_pred'] = (total_l_pred * (180/math.pi)).squeeze().numpy().tolist()
        results[l]['std_pred'] = (dists_l**2).mean().sqrt().item()
         
    results = sorted(results.items(), key=lambda x: x[1]['avg_dist'])
    with open(results_dir / f"results_per_lang_{args.suffix}.json", "w") as f:
        json.dump(results, f, indent=4) 

    macro_avg = sum(r[1]['avg_dist'] for r in results) / len(results)
    print(f"Macro_Acc: {macro_avg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--suffix", type=str, required=True)
    parser.add_argument("--cuts", type=str, required=True)
    args = parser.parse_args()
    main(args)
