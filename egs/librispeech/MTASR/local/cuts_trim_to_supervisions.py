from lhotse import load_manifest_lazy, CutSet
import argparse


def main(args):
    cuts = load_manifest_lazy(args.cuts)
    with CutSet.open_writer(args.cuts_trimmed) as cut_writer:
        for c in cuts.trim_to_supervisions(keep_overlapping=False, keep_all_channels=False):
            cut_writer.write(c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cuts")
    parser.add_argument("cuts_trimmed")
    args = parser.parse_args()
    main(args)


