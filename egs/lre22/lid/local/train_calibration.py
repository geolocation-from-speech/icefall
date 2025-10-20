import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import argparse


def main(args):
    scores = np.load(args.scores)
    tgts = np.load(args.tgts)
    langs = sorted(set(tgts))
    lang2int = {l: i for i, l in enumerate(langs)}
    tgts = np.array([lang2int[t] for t in tgts])

    Path(args.odir).mkdir(parents=True, exist_ok=True)
    # Create a logistic regression model for calibration
    calibration_model = Pipeline(
        [
            ('scaler', StandardScaler()),  # Normalize log-likelihoods for better calibration
            ('log_reg', LogisticRegression(solver='lbfgs', max_iter=1000))
        ]
    )

    import pdb; pdb.set_trace()
    # Train the model on labeled data
    calibration_model.fit(scores, tgts)
    joblib.dump(calibration_model, args.odir + "/cal.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores")
    parser.add_argument("--tgts")
    parser.add_argument("--odir")
    args = parser.parse_args()
    main(args)
