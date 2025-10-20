import numpy as np
from itertools import groupby 


min_val = 1e-20


def beta(P_target, C_FA=1, C_miss=1):
    return C_FA * (1-P_target) / (C_miss * P_target) 


def c_avg(beta: float, p_miss: np.ndarray, p_fa: np.ndarray):
    N_L = p_miss.shape[0]
    return (1./N_L) * (p_miss.sum() + (beta*p_fa.sum() / (N_L - 1)))
    

def c_primary(operating_points):
    vals = {o: c_avg(o, p_miss, p_fa) for p_miss, p_fa, o in operating_points}
    return sum(vals.values()) / len(vals), vals


def get_ratios(scores: np.ndarray):
    K = scores.shape[1]-1
    c = scores.max(axis=1)
    scores_ = scores - c[:, None]
    logsumexp = np.log(
        np.exp(scores_).sum(axis=1)[:, None] - np.exp(scores_) + min_val
    )
    LLRs = scores + np.log(K) - c[:, None] - logsumexp 
    return LLRs    


def compute_p_miss_p_fa(
    LLRs: np.ndarray,
    tgts: np.ndarray,
    beta=1.0,
):
    A = np.array(
        [
            np.array(
                [
                    (LLRs > np.log(beta))[tgts == j, i].sum() / sum(tgts == j)
                    for j in range(LLRs.shape[1])
                ]
            )
            for i in range(LLRs.shape[1])
        ]
    )
    import pdb; pdb.set_trace()
    p_miss = 1 - A.diagonal()
    mask = np.eye(A.shape[0], dtype=bool)
    p_fa = A[~mask].reshape(A.shape[0], A.shape[0]-1)
    return p_miss, p_fa, beta  


def get_groups(ids: np.ndarray):
    groups = {
        k: list(g)
        for k, g in groupby(
            sorted(zip(range(len(ids)), ids), key=lambda x: x[1]
        ), lambda x: x[1].split("-", 1)[0])
    }

    return groups


def group_field(field, groups, aggregate="mean"):
    if aggregate == "mean":
        grouped_field = np.array(
            [
                field[[i for i, _ in groups[k]]].mean(axis=0)
                for k in sorted(groups.keys())
            ]
        )

    if aggregate == "first":
        grouped_field = np.array(
            [
                field[[i for i, _ in groups[k]]][0]
                for k in sorted(groups.keys())
            ]
        )
    return grouped_field

