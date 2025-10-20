import argparse
from tqdm import tqdm

from lhotse import load_manifest_lazy, CutSet
from functools import reduce


def main(args):
    cuts = load_manifest_lazy(args.cuts)
    sps = [float(s) for s in args.speed_perturbs.split(",")]
    tps = [float(s) for s in args.tempo_perturbs.split(",")]
    windows = [float(w) for w in args.windows.split(",")]
    new_cuts = CutSet([])
    for sp in sps:
        new_cuts += cuts.perturb_speed(sp)

    for tp in tps:
        new_cuts += cuts.perturb_tempo(tp)

    new_cuts += cuts
    with CutSet.open_writer(args.out) as cut_writer:
        for c in tqdm(new_cuts, "Writing permuted ..."):
            for w in windows:
                c_windows = c.cut_into_windows(w)
                for p in range(args.num_permutes):
                    c_ = reduce(
                        lambda a, b: a.append(b),
                        list(c_windows.shuffle())
                    )
                    cut_writer.write(c_)
        for c in tqdm(new_cuts, "Writing original ..."):
            cut_writer.write(c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuts")
    parser.add_argument("--num-permutes", type=int, default=4)
    parser.add_argument("--windows", type=str, default="1,2,4", help="window sizes in seconds")
    parser.add_argument("--speed-perturbs", type=str, default="0.9,1.1", help="speed perturbation factors")
    parser.add_argument("--tempo-perturbs", type=str, default="0.8,1.2", help="tempo perturbation factors")
    parser.add_argument("--out")
    args = parser.parse_args()
    main(args)
