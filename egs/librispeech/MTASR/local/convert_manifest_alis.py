from lhotse import load_manifest_lazy
import json
import argparse
from tqdm import tqdm


def main(args):
    cuts = load_manifest_lazy(args.cuts)
    alignments = {}
    # To order speakers by start time, the segments have to appear in the
    # correct order.  
    if args.sort_strategy == "start_time": 
        cuts_start_times = [[s.start for s in c.supervisions] for c in cuts]
        sort_orders = [
            sorted(range(len(cst)), key=lambda i: cst[i])
            for cst in cuts_start_times
        ]
    # To order speakers by speaking time, the segments have to appear in a
    # special order. The segments need to be sorted by speaker and then the
    # the segments should be order by speaker, then start time
    elif args.sort_strategy == "speech_time":
        # Order speakers by speaking time
        sort_orders = []
        for c in cuts:
            speaker_durations = {}
            for s in c.supervisions:
                if s.speaker not in speaker_durations:
                    speaker_durations[s.speaker] = 0
                speaker_durations[s.speaker] += s.duration
            
            sorted_speaker_durations = sorted(
                speaker_durations.items(),
                key=lambda x: x[1],
                reverse=True
            )

            sort_order = []
            for spk, dur in sorted_speaker_durations:
                for i, s in enumerate(c.supervisions):
                    if s.speaker == spk:
                        sort_order.append(i)
            sort_orders.append(sort_order)
    
    for c_idx, c in tqdm(enumerate(cuts)):
        for sort_idx, i in enumerate(sort_orders[c_idx]):
            s = cuts[c_idx].supervisions[i]
            spk = s.speaker
            utt_spk = []
            for a in s.alignment["word"]:
                if a.symbol.strip() == "":
                    continue
                utt_spk.append([spk, sort_idx, a.symbol, a.start + s.start, a.duration])
            if c.id not in alignments:
                alignments[c.id] = []
            alignments[c.id].extend(utt_spk)
    for k in alignments:
        alignments[k] = sorted(alignments[k], key=lambda x: (x[0], x[3]))
    alignments = sorted(alignments.items(), key=lambda x: x[0])
    with open(args.ali_out, 'w') as f:
        json.dump(alignments, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cuts")
    parser.add_argument("ali_out")
    parser.add_argument("--sort-strategy", type=str, default="start_time")
    args = parser.parse_args()
    main(args)
