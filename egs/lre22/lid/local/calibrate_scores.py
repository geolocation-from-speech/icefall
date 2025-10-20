import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import argparse


def main(args):
    import pdb; pdb.set_trace()
    mdl = joblib.load(args.mdl)
    scores = np.load(args.scores)
    new_scores = mdl.predict_log_proba(scores)
    np.save(args.scores_cal, new_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores")
    parser.add_argument("--mdl")
    parser.add_argument("--scores-cal")
    args = parser.parse_args()
    main(args)
