import argparse
import json
from pathlib import Path


def segs2word(segs):
    word = "".join([a[2] for a in segs]).replace("\u2581", "")
    start = segs[0][3]
    # start + dur of last segment
    end = segs[-1][3] + segs[-1][4]
    dur = end - start
    return segs[0][0], segs[0][1], word, start, dur



def main(args):
    alis = json.load(open(args.alis))
    word_alis = []
    for utt in alis:
        utt_ali = sorted(utt[1], key=lambda x: (x[1], x[3]))
        curr_word = [utt_ali[0]]
        new_utt = []
        for spk, seg, token, start, dur in utt_ali[1:]:
            # Check that the token is the beginning of sequence token and flush
            # the current word
            if token.startswith("\u2581"):
                # Construct the current word
                spk_w, seg_w, w, start_w, dur_w = segs2word(curr_word)
                if dur_w <= 0:
                    import pdb; pdb.set_trace()
                assert dur_w > 0
                new_word = [spk_w, seg_w, w, round(start_w, 3), round(dur_w,3)]

                # We need to figure out to which speaker to add the new word
                new_utt.append(new_word)
                curr_word = []
            curr_word.append([spk, seg, token, start, dur])
        
        # Flush the last word at the end of the loop
        if len(curr_word) > 0:
            spk_w, seg_w, w, start_w, dur_w = segs2word(curr_word)
            if dur_w <= 0:
                import pdb; pdb.set_trace()
            assert dur_w > 0
            new_word = [spk_w, seg_w, w, round(start_w, 3), round(dur_w, 3)]
            new_utt.append(new_word)
             
        # Append utterance to alignment list
        word_alis.append([utt[0], new_utt])
    
    name = Path(args.alis).stem
    parent = Path(args.alis).parent
    with open(parent / f"word_{name}.json", "w") as f:
        json.dump(word_alis, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("alis", type=str)
    args = parser.parse_args()
    main(args)
