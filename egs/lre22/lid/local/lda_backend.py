import numpy as np
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.lda import LDA
import joblib


def main(args):
    embeds = np.load(args.embeds)
    mean = np.mean(embeds, axis=0)
    embeds -= mean

    tgts = np.load(args.tgts)
    unique_tgts = np.unique(tgts, axis=0)
    lda = LDA(n_components=len(unique_tgts))
    lda.fit(embeds, tgts)
    if args.out is None:
        modeldir = Path(args.embeds).parent / "lda_backend"
        outfile = modeldir / "mdl"    
    else:
        outfile = Path(args.out)  
    outfile.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(lda, f'{outfile}.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Convert numpy embeddings to scores assuming a Gaussian model for "
        "each class with Identity covariance."
    )
    parser.add_argument("--embeds", type=str, required=True)
    parser.add_argument("--tgts", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--norm", type=str, default="True")
    args = parser.parse_args()
    main(args)
