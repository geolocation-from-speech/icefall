import argparse
from pathlib import Path
from lhotse import load_manifest_lazy
from tqdm import tqdm
from lhotse.parallel import parallel_map
from lhotse import MonoCut
from lhotse import CutSet
from functools import partial
from lhotse.utils import fastcopy


def fix_text(c, noises_dict, sorted_noises):
    new_text = c.supervisions[0].text
    for n in sorted_noises:
        new_text = new_text.replace(n, noises_dict[n])  
    c_ = fastcopy(c) 
    c_.supervisions[0].text = new_text
    return c_


def main(args):
    noises_dict = {}
    with open(args.noise_ids) as f:
        for l in f:
            noise_id, noise_str = l.strip().split(None, 1)
            if noise_str not in noises_dict:
                noises_dict[noise_str] = noise_id
    
    sorted_noises = sorted(noises_dict, key=lambda n: len(n), reverse=True)
    
    fun = partial(fix_text, sorted_noises=sorted_noises, noises_dict=noises_dict) 
    cuts = load_manifest_lazy(args.cuts) 
    with CutSet.open_writer(args.output) as writer:
        for c in tqdm(
            parallel_map(fun, cuts, num_jobs=args.num_jobs),
        ):
            writer.write(c)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("noise_ids")
    parser.add_argument("cuts")
    parser.add_argument("output")
    parser.add_argument("--num-jobs", type=int, default=4)
    args = parser.parse_args()
    main(args)
