from __future__ import annotations
import argparse
import json

from typing import Any, Dict, Iterable, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


TokenRow = List[Any]  # [speaker, segment, token, start, duration]


def _check_segment_speaker_consistency(rows: List[TokenRow], name: str) -> Dict[Any, Any]:
    """
    Ensure each segment has a single speaker label. Returns segment->speaker.
    """
    seg2spk: Dict[Any, Any] = {}
    for spk, seg, *_ in rows:
        if seg not in seg2spk:
            seg2spk[seg] = spk
        elif seg2spk[seg] != spk:
            raise ValueError(
                f"{name}: segment {seg} has inconsistent speaker labels: "
                f"{seg2spk[seg]} vs {spk}"
            )
    return seg2spk


def hungarian_mapping_token_cost(
    pred_rows: List[TokenRow],
    ref_rows: List[TokenRow],
    *,
    align_by_index: bool = True,
    join_key: Optional[str] = None,
) -> Dict[Any, Any]:
    """
    Compute best permutation pred_speaker -> ref_speaker using TOKEN-LEVEL counts.

    Assumes pred_rows and ref_rows refer to the same token instances.
    By default it assumes they are aligned by list index.

    If you need joining, set align_by_index=False and implement a join key
    strategy (see notes below).
    """
    if align_by_index:
        if len(pred_rows) != len(ref_rows):
            raise ValueError(
                f"Token lists differ in length ({len(pred_rows)} vs {len(ref_rows)}). "
                "If they aren't aligned by index, you need a join key."
            )
        pred_spk_seq = [r[0] for r in pred_rows]
        ref_spk_seq = [r[0] for r in ref_rows]
    else:
        raise NotImplementedError(
            "Join-key alignment not implemented in this snippet. "
            "Tell me what key uniquely identifies a token instance and I’ll add it."
        )

    pred_labels = list(dict.fromkeys(pred_spk_seq))  # preserve order
    ref_labels  = list(dict.fromkeys(ref_spk_seq))

    P, R = len(pred_labels), len(ref_labels)
    n = max(P, R)

    pred_idx = {lab: i for i, lab in enumerate(pred_labels)}
    ref_idx  = {lab: j for j, lab in enumerate(ref_labels)}

    # Token-level confusion / agreement
    A = np.zeros((P, R), dtype=np.int64)
    for p, r in zip(pred_spk_seq, ref_spk_seq):
        A[pred_idx[p], ref_idx[r]] += 1

    # Pad to square
    A_pad = np.zeros((n, n), dtype=np.int64)
    A_pad[:P, :R] = A

    # Hungarian solves min-cost; we want max agreement => cost = -agreement
    cost = -A_pad
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping: Dict[Any, Any] = {}
    for i, j in zip(row_ind, col_ind):
        if i < P:
            p_lab = pred_labels[i]
            mapping[p_lab] = ref_labels[j] if j < R else None  # None = dummy/unmatched

    return mapping


def apply_mapping_at_segment_level(
    pred_rows: List[TokenRow],
    mapping: Dict[Any, Any],
    *,
    unmapped_sentinel: Any = None,
) -> List[TokenRow]:
    """
    Apply pred speaker -> ref speaker mapping, but conceptually at SEGMENT level.
    (Since speaker is constant per segment, applying per row is equivalent.)
    """
    # sanity: segments should be single-speaker
    _check_segment_speaker_consistency(pred_rows, "pred")

    out: List[TokenRow] = []
    for spk, seg, tok, start, dur in pred_rows:
        new_spk = mapping.get(spk, spk)
        if new_spk is None:
            new_spk = unmapped_sentinel if unmapped_sentinel is not None else spk
        out.append([new_spk, seg, tok, start, dur])
    return out


def token_speaker_error(pred_rows: List[TokenRow], ref_rows: List[TokenRow]) -> int:
    """
    Token-level speaker mismatch count (assumes rows are aligned token-by-token).
    """
    if len(pred_rows) != len(ref_rows):
        raise ValueError("pred_rows and ref_rows must have same length for token error.")
    return sum(1 for pr, rr in zip(pred_rows, ref_rows) if pr[0] != rr[0])


   
    

def main(args):
    hyp = json.load(open(args.hyp))
    ref_ = json.load(open(args.ref))
    bias = args.bias / 1000
    duration_bias = args.bias / 1000
    hyp_utts = [h[0] for h in hyp]
    ref = []
    skipped = 0
    total = 0
    for r in ref_:
        if r[0] not in hyp_utts:
            #print(f"Removing {r[0]} from evaluation ...")
            skipped += 1
        else:
            ref.append(r)
        total += 1 
    total_intersection = 0
    total_union = 0
    ious = []
    total_errors = []
    boundary_errors = []
    duration_errors = []
    speaker_errors = 0
    total_tokens = 0
    num_utts = len(ref)
    partition_errors = 0
    calibrate_set = num_utts // 2
    if args.reverse:
        hyp = hyp[::-1]
        ref = ref[::-1]
    for i, (utt_hyp, utt_ref) in tqdm(enumerate(zip(hyp, ref))):
        if i == calibrate_set and args.calibrate:
            bias = sum(boundary_errors) / len(boundary_errors)
            duration_bias = sum(duration_errors) / len(duration_errors)
            total_errors = []
            ious = []
            boundary_errors = []
            duration_errors = []
            total_intersection = 0
            total_union = 0

        h = sorted(utt_hyp[1], key=lambda x: (x[1], x[3]))
        r = sorted(utt_ref[1], key=lambda x: (x[1], x[3]))
        assert len(h) == len(r)
        
        # Here we are going to relabel the speakers as restricted growth frames
       
        if args.relabel:
            mapping = hungarian_mapping_token_cost(h, r)
            h_ = apply_mapping_at_segment_level(h, mapping)
        else:
            h_ = h

        utt_errors = []
        utt_spk_errors = False
        for h_i, r_i in zip(h_, r):
            h_s, r_s = h_i[3] - bias, r_i[3]
            h_e, r_e = h_s + h_i[4] - duration_bias, r_s + r_i[4]
            #h_e, r_e = h_s + h_i[4], r_s + r_i[4]
            
            utt_errors.append(0.5*(abs(h_s - r_s) + abs(h_e - r_e)))
            boundary_errors.append(h_s - r_s)
            duration_errors.append((h_e - h_s) - (r_e - r_s))
            union = max(h_e, r_e) - min(h_s, r_s)
            intersection = max(0, min(h_e, r_e) - max(h_s, r_s))
            assert intersection <= union
            ious.append(intersection / union)
            #if h_i[0] != r_i[0]:
            #    import pdb; pdb.set_trace()
            speaker_errors += (h_i[0] != r_i[0])
            utt_spk_errors = utt_spk_errors or (h_i[0] != r_i[0])
            total_tokens += 1
            total_intersection += intersection
            total_union += union
        partition_errors += int(utt_spk_errors)
        total_errors.append(sum(utt_errors)/len(utt_errors))
    print(f"Mean IoU: {100*sum(ious) / len(ious):0.1f} over {len(ious)} segments")
    print(f"IoU: {100*total_intersection/total_union:0.1f}")
    print(f"Mean Boundary Error: {int(1000*sum(total_errors) / len(total_errors))} ms")
    print(f"Mean start error: {int(1000 * sum(boundary_errors) / len(boundary_errors))} ms")
    print(f"Mean duration error: {int(1000 * sum(duration_errors) / len(duration_errors))} ms")
    print(f"Failed on: {100*skipped / total}%")
    print(f"WDER: {100*speaker_errors / total_tokens}%") 
    print(f"Partition Errors: {100*partition_errors / len(total_errors)}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hyp", "-h", type=str, required=True)
    parser.add_argument("--ref", "-r", type=str, required=True)
    parser.add_argument("--bias", "-b", type=float, default=0.0)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--relabel", action="store_true")
    parser.add_argument("--help", action="help")
    args = parser.parse_args()
    main(args)
