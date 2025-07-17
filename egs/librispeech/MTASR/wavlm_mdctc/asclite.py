from mdctc_graph_compiler2 import MDCTCGraphCompiler
import k2


def add_ins_del_sub_arcs(fsavec: k2.Fsa, new_score=-1.0, arc_types=("ins", "del", "loop", "sub"), vocab=None) -> k2.Fsa:
    new_fsas = []
    for i in range(fsavec.shape[0]):
        fsa = fsavec[i]
        new_arcs = set()
        arcs = k2.to_str(fsa).strip().split("\n")
        for a in arcs:
            new_arcs.add(a)
            vals = a.split()
            if len(vals) > 2 and int(vals[2]) != -1:
                src, dest, lbl, aux_lbl, score = vals
                src, dest, lbl, aux_lbl, score = int(src), int(dest), int(lbl), int(aux_lbl), float(score)
                if "del" in arc_types:
                    new_arcs.add(f"{src} {dest} 0 {lbl} {new_score}")
                if "ins" in arc_types:
                    new_arcs.add(f"{src} {dest} {lbl} 0 {new_score}")
                if "loop" in arc_types:
                    new_arcs.add(f"{src} {src} 0 0 {new_score}")
                if "sub" in arc_types:
                    if isinstance(vocab, set):
                        for v in vocab:
                            if v != lbl and v != -1:
                                new_arcs.add(f"{src} {dest} {v} {lbl} {new_score}")
                    elif isinstance(vocab, list):
                        for v in vocab[i]:
                            if v != lbl and v != -1:
                                new_arcs.add(f"{src} {dest} {v} {lbl} {new_score}")
        new_arcs = sorted(new_arcs, key=lambda x: int(x.split()[0]))
        new_fsa = k2.Fsa.from_str("\n".join(new_arcs), acceptor=False)
        new_fsa = k2.arc_sort(new_fsa)
        new_fsas.append(new_fsa)
    return k2.create_fsa_vec(new_fsas)


def compute_ter(decoding_graphs, hyp):
    hyp = k2.linear_fst(hyp, hyp)
    assert decoding_graphs.shape[0] == hyp.shape[0]
    hyp = add_ins_del_sub_arcs(hyp, arc_types=("ins", "loop"))
    hyp = k2.arc_sort(hyp)
    
    vocab = []
    for i in range(decoding_graphs.shape[0]):
        vocab.append(
            set(decoding_graphs[i].labels.tolist() + hyp[i].labels.tolist())
        )
    
    decoding_graphs.aux_labels = decoding_graphs.labels.clone()
    decoding_graphs = add_ins_del_sub_arcs(
        decoding_graphs,
        arc_types=("del", "loop", "sub"),
        vocab=vocab
    )
    decoding_graphs = k2.arc_sort(decoding_graphs)
    graphs = k2.compose(hyp, decoding_graphs, treat_epsilons_specially=True)
    graphs = k2.connect(graphs)
    graphs = k2.remove_epsilon_self_loops(graphs)
    graphs = k2.arc_sort(graphs)
    graphs = k2.connect(graphs)
    best_paths = k2.shortest_path(graphs, use_double_scores=True)
    errors = -best_paths.scores.sum()
    total = best_paths.scores.size(0)
    dels = (best_paths.labels == 0).sum()
    ins  = (best_paths.aux_labels == 0).sum()
    corr = (best_paths.labels[:-1] == best_paths.aux_labels[:-1]).sum()
    subs = (
        (best_paths.labels[:-1] != best_paths.aux_labels[:-1]).sum()
        - dels - ins
    ) 
    alignments = []
    for i in range(best_paths.shape[0]):
        ali_hyp, ali_ref = [], []
        for l_hyp, l_ref in zip(best_paths[i].labels[:-1], best_paths[i].aux_labels[:-1]):
            ali_hyp.append(l_hyp.item())
            ali_ref.append(l_ref.item())
        alignments.append([ali_hyp, ali_ref])
    return errors.item(), total, (errors/total).item(), alignments, dels.item(), ins.item(), subs.item(), corr.item()
