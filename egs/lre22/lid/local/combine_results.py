import numpy as np
import argparse
from pathlib import Path
from itertools import groupby


def main(args):
    resultsdir = Path(args.results_dir)
    embeds_files = sorted(resultsdir.rglob(f"embeds_{args.suffix}*"), key=lambda x: x.stem.split("_")[0])
    preds_files = sorted(resultsdir.rglob(f"preds_{args.suffix}*"), key=lambda x: x.stem.split("_")[0])
    tgts_files = sorted(resultsdir.rglob(f"tgts_{args.suffix}*"), key=lambda x: x.stem.split("_")[0])
    ids_files = sorted(resultsdir.rglob(f"ids_{args.suffix}*"), key=lambda x: x.stem.split("_")[0])
    scores_files = sorted(resultsdir.rglob(f"scores_{args.suffix}*"), key=lambda x: x.stem.split("_")[0])
    
    for files_type in [embeds_files, preds_files, tgts_files, ids_files, scores_files]:
        if len(files_type) > 1:
            ftype = files_type[0].stem.split("_")[0]
            new_obj = np.concatenate([np.load(f) for f in files_type], axis=0)
            np.save(resultsdir / f"{ftype}_{args.suffix}.npy", new_obj)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Combine numpy embeddings across splits"
    )
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--suffix", type=str, required=True)
    args = parser.parse_args()
    main(args)
