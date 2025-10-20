import argparse
from pathlib import Path
import numpy as np
from itertools import groupby 
from tqdm import tqdm
from LREScorer import * 


def main(args):
    # Use 2 different cost functions (p_target=0.5, p_target=0.1)
    p_tgts = [float(p) for p in args.p_tgts.split(",")]
    
    # load the scores
    scores = np.load(args.scores)

    # The scores might be over smaller segments instead of over the full
    # utterance. If ids are provided, these can be used to aggregate the
    # segment-level scores into utterance level scores
    if args.ids is not None:
        ids = np.load(args.ids)

    # Load the targets
    tgts = np.load(args.tgts)
    langs = sorted(set(tgts))
    lang2int = {l:i for i, l in enumerate(langs)}
    int2lang = {i: l for l, i in lang2int.items()}
    
    # Integerize them
    tgts = np.array([lang2int[l] for l in tgts]) 
    
   
    import pdb; pdb.set_trace() 
    # Group things based on the ids
    if args.ids is not None:
        groups = get_groups(ids)
        scores = group_field(scores, groups, aggregate="mean")
        tgts = group_field(tgts, groups, aggregate="first")
        if args.nist is not None:
            with open(args.nist, "w") as f:
                print(f"segmentid\t" + "\t".join(langs), file=f)
                tab = "\t"
                for i, uttid in enumerate(sorted(groups.keys())):
                    print(
                        f"{uttid}\t{tab.join([str(s) for s in scores[i]])}",
                        file=f
                    )

    
    # Get the log-likelihood ratios
    LLRs = get_ratios(scores)

    # For both operating points, compute the p_miss and p_false_alarm
    vals = [compute_p_miss_p_fa(LLRs, tgts, beta(p)) for p in p_tgts]
   
    # Compute the cost function as defined in the LRE22 evaluation plan
    c_act, c_act_dict = c_primary(vals) 
    print(f"Cpact: {c_act}")
    print(f"Cpact: {c_act_dict}")
   
    # I don't think this next part is implemented correctly, or works yet. 
    if not args.skip_cpmin:
        best_val = c_act
        best_thresh = 1.0
        val_dict = c_act_dict
        for v in tqdm(np.logspace(-3, 3, num=200), "Sweeping ..."):
            p_miss, p_fa, _ = compute_p_miss_p_fa(LLRs, tgts, v)
            result = c_primary([[p_miss, p_fa, 1.0], [p_miss, p_fa, 9.0]])  
            if result[0] < best_val:
                best_val = result[0]
                val_dict = result[1]
                best_thresh = np.log(v)
         
        print(f"Cpmin {best_val} ({best_thresh}))")
        print(f"Cpmin: {val_dict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("A script for LRE22 scoring")
    parser.add_argument("--scores",
        type=str,
        help="Path to a numpy array of scores for each segment. The scores are"
        " interpretted as log-likelihoods for each language",
        required=True,
    )
    parser.add_argument("--tgts",
        type=str,
        help="Path to a numpy array of the groundtruth for each segment.",
        required=True,
    )
    parser.add_argument("--ids",
        type=str,
        help="Path to numpy array of ids",
    )
    parser.add_argument("--p-tgts", type=str, default="0.5,0.45,0.4,0.3,0.2,0.1")
    parser.add_argument("--skip-cpmin", action="store_true")
    parser.add_argument("--nist", type=str, default=None)
    args = parser.parse_args()
    main(args) 
