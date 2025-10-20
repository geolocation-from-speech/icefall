import numpy as np
import argparse
from pathlib import Path

# (x - mu)^T P (x - mu)
# (x - mu)^T(Px - Pmu) = x^T P x - mu^T P x + mu^T P mu - x^T P mu
#                         Nxd dxd dxN  - 1xd dxd dx1     1xd dxd dx1     Nxd dxd dx1
#                    
#                        x^T (P (

def compute_ll(x, mu, P):
    X_centered = (x[:, :, None] - mu[:, None, :])
    X_centered_transformed = np.tensordot(P, X_centered, axes=(0, 0))
    ll = -np.einsum('dni,dni->ni', X_centered, X_centered_transformed) / 2
    return ll


def main(args):
    embeds = np.load(args.embeds)
    mdl = np.load(args.backend, allow_pickle=True)
    langs = sorted(set(mdl.keys()).difference({"cov", "mean", "U", "norm"}))
    means_ = []
    for l in langs:
        means_.append(mdl[l])
    means = np.array(means_)
    cov = mdl["cov"] 
    global_mean = mdl["mean"]

    prec = np.linalg.inv(cov)
  
    import pdb; pdb.set_trace()
    if args.test_mean:
        global_mean = embeds.mean(axis=0) 
    # Subtract global mean and do PCA (project to low dimension)
    embeds -= global_mean
    embeds = embeds @ mdl["U"]
    if mdl["norm"] == 1:
        row_norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds /= row_norms
    embeds = embeds.T
    means = means.T
    ll = compute_ll(embeds, means, prec)  
   
    if args.out is None:
        outfile = Path(args.backend).parent / ("scores_" + str(Path(args.embeds).stem))
    else:
        outfile = Path(args.out)  
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.save(outfile, ll)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Convert numpy embeddings to scores assuming a Gaussian model for "
        "each class with Identity covariance."
    )
    parser.add_argument("--embeds", type=str, required=True)
    parser.add_argument("--backend", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--test-mean", action="store_true")
    args = parser.parse_args()
    main(args)
