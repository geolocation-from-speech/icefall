import numpy as np
import argparse
from pathlib import Path


def main(args):
    embeds = np.load(args.embeds)
    tgts = np.load(args.tgts)
    langs = sorted(set(tgts))
    params = {}       
    covs = []
    if args.pca_dim > 0:
        mean = embeds.mean(axis=0)
        embeds -= mean
        U, S, Vt = np.linalg.svd(embeds.T, full_matrices=False)
        embeds = embeds @ U[:, :args.pca_dim]
        if args.norm == "True":
            row_norms = np.linalg.norm(embeds, axis=1, keepdims=True)
            embeds /= row_norms
              
    for l in langs:
        m_l = embeds[tgts == l].mean(axis=0)
        cov_l = np.cov(embeds[tgts == l], rowvar=False)
        covs.append(cov_l) 
        params[l] = m_l
    
    if args.cov_type == "global":
        cov = np.cov(embeds, rowvar=False)
        params["cov"] = cov
    else:
        cov = np.array(covs).mean(axis=0) 
        params["cov"] = cov
    params["mean"] = mean
    params["U"] = U[:, :args.pca_dim] 
    params["norm"] = 1 if args.norm == "True" else 0
    if args.out is None:
        modeldir = Path(args.embeds).parent / "gaussian_backend"
        outfile = modeldir / "mdl"    
    else:
        outfile = Path(args.out)  
    outfile.parent.mkdir(parents=True, exist_ok=True)

    np.savez(outfile, **params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Convert numpy embeddings to scores assuming a Gaussian model for "
        "each class with Identity covariance."
    )
    parser.add_argument("--embeds", type=str, required=True)
    parser.add_argument("--tgts", type=str, required=True)
    parser.add_argument("--pca-dim", type=int, default=0)
    parser.add_argument("--cov-type", type=str, default="per_lang", choices=["per_lang", "global"])
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--norm", type=str, default="True")
    args = parser.parse_args()
    main(args)
