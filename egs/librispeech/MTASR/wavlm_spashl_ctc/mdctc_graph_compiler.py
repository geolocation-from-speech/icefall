#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Author: Matthew Wiesner
# Year: 2025

import argparse
import k2
import sentencepiece as spm
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict
import torch
from collections import defaultdict, deque
from itertools import groupby
import logging


class MDCTCGraphCompiler(object):
    """
    A graph compiler for multi-dimensional CTC decoding with token-level timing constraints.

    This class is designed to construct decoding graphs (FSAs) for use in alignment or training
    under the Multi-Dimensional CTC (MD-CTC) framework. It uses a SentencePiece model to map
    transcripts to token sequences and can encode optional time-collar constraints between tokens.

    Attributes
    ----------
    sp : sentencepiece.SentencePieceProcessor
        The SentencePiece model used to tokenize input strings.
    device : Union[str, torch.device]
        The device on which FSAs will be created and processed (e.g., "cpu" or "cuda").
    collar : int
        A temporal padding (in frames or milliseconds) to apply as a constraint margin between tokens.
    """
    def __init__(
        self,
        lang_dir: Path,
        device: Union[str, torch.device] = "cpu",
    ):
        """
            Initialize the MDCTCGraphCompiler.

            Loads the SentencePiece model from the given language directory and sets up
            internal parameters such as device and collar width.

            :param lang_dir: Path to the language directory containing the `bpe.model` file.
            :type lang_dir: Path
            :param device: The device to use for FSA operations (e.g., "cpu" or "cuda").
            :type device: Union[str, torch.device]
            :param collar: Temporal collar size (e.g., in milliseconds or frames) to allow flexibility in alignment.
            :type collar: int
        """
        self.device = device
        # Create a map from (speaker, token) tuples --> int
        lang_dir = Path(lang_dir)
        bpe_model_file = lang_dir / "bpe.model"
        sp = spm.SentencePieceProcessor() 
        sp.load(str(bpe_model_file))
        self.sp = sp


    def build_ctc_topo(self, symbols: List[int], original=False) -> k2.Fsa:
        """
            Builds a CTC topology over the provided symbols only.

            :param symbols: The list of the symbols over which to build the 
                            ctc topology. 0 is assumed to be blank on the input
                            and epsilon on the output.
        """
        assert 0 not in symbols, "Symbol 0 is reserved for the blank token in CTC."
        
        symbols = list(set(symbols))
        arcs = []
        blank_state = 0  # Will set this properly later
        arcs.append(f"{blank_state} {blank_state} 0 0 0.0")  # self-loop with blank
        arcs.append(f"{blank_state} {len(symbols)+1} -1 -1 0.0")
        for i, s1 in enumerate(symbols, 1):
            # Blank state:
            #  - self-loop with blank
            #  - arc to symbol state with symbol label
            arcs.append(f"{blank_state} {i} {s1} {s1} 0.0")    # to symbol state
        
            # only self-loop and go to blank
            arcs.append(f"{i} 0 0 0 0.0")  # to blank
            if original:
                arcs.append(f"{i} {i} {s1} 0 0.0")     # self-loop with symbol
                for j, s2 in enumerate(symbols, 1):
                    if i == j:
                        continue
                    arcs.append(f"{i} {j} {s2} {s2} 0.0")
            arcs.append(f"{i} {len(symbols)+1} -1 -1 0.0")
        arcs = sorted(arcs, key=lambda x: int(x.split()[0]))
        arcs.append(f"{len(symbols)+1}")
        # Build arc list for k2
        fsa = k2.Fsa.from_str("\n".join(arcs), acceptor=False)
        
        fsa = k2.arc_sort(fsa)
        return fsa 

    def get_seqs_and_constraints(
        self,
        c: List[str],
        offsets: List[int],
        lens: List[int],
        spks: List[int],
        spk2int: Dict,
        collar=64000,
        allow_self_overlap: bool = True,
    ) -> Tuple[List[List[str]], List[Tuple[str, str]], Dict[str, int]]:
        """
            Generate a list of token-level labels and precedence constraints for a sequence.
    
            This method tokenizes each string in `c` using the SentencePiece model,
            associates each token with an estimated start time based on `offsets` and `lens`,
            and returns structured data useful for constructing topologically sorted FSAs
            with timing-aware constraints.
    
            Each token is annotated with a position-specific label (e.g., `'532_0_1'`), where
            `532` is the token ID, `0` is the string index, and `1` is the token index within the string.
            Constraints are created between tokens across sequences based on their relative start times.
    
            :param c: A list of strings (e.g., words or phrases) to be tokenized.
            :type c: List[str]
            :param offsets: Start time for each string in `c`, typically in frames or milliseconds.
            :type offsets: List[int]
            :param lens: Duration (in time units) of each string in `c`.
            :type lens: List[int]
    
            :returns: A tuple containing:
                - seqs (List[List[str]]): Token labels for each input string, with unique positional suffixes.
                - constraints (List[Tuple[str, str]]): List of precedence constraints between tokens.
                - sym_map (Dict[str, int]): Mapping from positional token labels to original SentencePiece token IDs.
            :rtype: Tuple[List[List[str]], List[Tuple[str, str]], Dict[str, int]
        """        
        constraints = []
        # Find the total len of the sequence
        length = max([o + l for o, l in zip(offsets, lens)])

        # Convert token index to time index (i.e., in terms of frame offsets)
        token_start_times = {}
        pos_sym_to_sym = {}
        seqs = []
        for i, (s, o, l, spk) in enumerate(zip(self.sp.encode(c), offsets, lens, spks)):
            if len(s) == 0:
                continue
            samples_per_token  = l // len(s) + 1
            labels = []
            for j, token in enumerate(s):
                pos_sym = (token, spk2int[spk], j, i)
                pos_sym_to_sym[pos_sym] = token
                token_start_times[pos_sym] = o + j*samples_per_token
                labels.append(pos_sym)
            seqs.append(labels)
       
        num_constraints = 0
        num_ovlps = 0  
        for ki in token_start_times:
            for kj in token_start_times:
                # Only consider cross sequence comparisons
                if ki[3] == kj[3]:
                    continue
                diff = token_start_times[ki] - token_start_times[kj]
                # hard constraints for different utterances by the same speaker
                # or utterances that are far away
                far_enough = abs(diff) > collar # or ki[1] == kj[1] 
                if not allow_self_overlap:
                    far_enough = far_enough or (ki[1] == kj[1])
                if far_enough:
                    num_constraints += 1
                else:
                    num_ovlps += 1
                if far_enough and diff < 0:
                    constraints.append((ki, kj))
                elif far_enough:
                    constraints.append((kj, ki))
        return seqs, constraints, num_ovlps, pos_sym_to_sym

    def get_seqs_and_constraints_no_spk(
        self,
        c: List[str],
        offsets: List[int],
        lens: List[int],
        collar=64000,
        allow_self_overlap: bool = False,
    ) -> Tuple[List[List[str]], List[Tuple[str, str]], Dict[str, int]]:
        """
            Generate a list of token-level labels and precedence constraints for a sequence.
    
            This method tokenizes each string in `c` using the SentencePiece model,
            associates each token with an estimated start time based on `offsets` and `lens`,
            and returns structured data useful for constructing topologically sorted FSAs
            with timing-aware constraints.
    
            Each token is annotated with a position-specific label (e.g., `'532_0_1'`), where
            `532` is the token ID, `0` is the string index, and `1` is the token index within the string.
            Constraints are created between tokens across sequences based on their relative start times.
    
            :param c: A list of strings (e.g., words or phrases) to be tokenized.
            :type c: List[str]
            :param offsets: Start time for each string in `c`, typically in frames or milliseconds.
            :type offsets: List[int]
            :param lens: Duration (in time units) of each string in `c`.
            :type lens: List[int]
    
            :returns: A tuple containing:
                - seqs (List[List[str]]): Token labels for each input string, with unique positional suffixes.
                - constraints (List[Tuple[str, str]]): List of precedence constraints between tokens.
                - sym_map (Dict[str, int]): Mapping from positional token labels to original SentencePiece token IDs.
            :rtype: Tuple[List[List[str]], List[Tuple[str, str]], Dict[str, int]
        """        
        constraints = []
        # Find the total len of the sequence
        length = max([o + l for o, l in zip(offsets, lens)])

        # Convert token index to time index (i.e., in terms of frame offsets)
        token_start_times = {}
        pos_sym_to_sym = {}
        seqs = []
        for i, (s, o, l) in enumerate(zip(self.sp.encode(c), offsets, lens)):
            if len(s) == 0:
                continue
            samples_per_token  = l // len(s) + 1
            labels = []
            for j, token in enumerate(s):
                pos_sym = (token, j, i)
                pos_sym_to_sym[pos_sym] = token
                token_start_times[pos_sym] = o + j*samples_per_token
                labels.append(pos_sym)
            seqs.append(labels)
       
        num_constraints = 0
        num_ovlps = 0  
        for ki in token_start_times:
            for kj in token_start_times:
                # Only consider cross sequence comparisons
                if ki[2] == kj[2]:
                    continue
                diff = token_start_times[ki] - token_start_times[kj]
                # hard constraints for different utterances by the same speaker
                # or utterances that are far away
                far_enough = abs(diff) > collar # or ki[1] == kj[1] 
                if not allow_self_overlap:
                    far_enough = far_enough or (ki[1] == kj[1])
                if far_enough:
                    num_constraints += 1
                else:
                    num_ovlps += 1
                if far_enough and diff < 0:
                    constraints.append((ki, kj))
                elif far_enough:
                    constraints.append((kj, ki))
        return seqs, constraints, num_ovlps, pos_sym_to_sym

    
    def compile(
        self,
        cuts: List[List[str]],
        offsets,
        lens,
        spks,
        collar: int = 64000,
        max_overlaps: int = 6000, 
        original_topo: bool = False,
        allow_self_overlap: bool = False,
        debug: bool = False,
    ) -> k2.Fsa:
        """
            Compile a batch of transcripts into CTC-constrained decoding graphs.

            This function encodes each transcript into subword/token IDs, constructs a
            constrained alignment FSA based on per-token constraints, and composes it
            with a CTC topology graph to produce a training graph. The result is a
            `k2.FsaVec` of decoding graphs suitable for CTC training.

            :param cuts: A batch of transcripts, where each transcript is a list of string tokens.
            :type cuts: List[List[str]]
            :param offsets: A batch of starting offsets for each token in each transcript.
            :type offsets: List[List[int]] or compatible structure
            :param lens: A batch of lengths (e.g., in frames or tokens) for each token in each transcript.
            :type lens: List[List[int]] or compatible structure
            :param spks: A batch of lists of speaker labels
            :param debug: Flag for debugging outputs
            :type debug: bool
            :param max_overlaps: the maximum number of overlapping tokens allowed
            :type max_overlaps: int
            :param original_topo: Which ctc topology to use
            :type original_topo: bool
            :param allow_self_overlap: multiple utterances by the same speaker
                can overlap
            :type allow_self_overlap: bool
            :return: A batched FSA (`FsaVec`) representing all input transcripts composed with a CTC topology.
            :rtype: k2.Fsa
        """
        graphs = []
        for c, o, l, spk in zip(cuts, offsets, lens, spks):
            # This line is critical. The supervisions are ordered in some
            # specified way, in the list spk. There may be duplicate elements
            # in the list. dict.fromkeys() preserves the ordering and removes
            # duplicates. This feature was introduced in python 3.7.
            spk2int = {k: i for i, k in enumerate(dict.fromkeys(spk))}
            # Estimate
            num_ovlps = max_overlaps + 1
            collar_ = collar
            seqs, constraints, num_ovlps, sym_map = self.get_seqs_and_constraints(
                c, o, l, spk, spk2int,
                collar=collar_,
                allow_self_overlap=allow_self_overlap,
            )
            while num_ovlps > max_overlaps: 
                collar_ = collar_ // 2
                logging.info(f"Number of overlaps {num_ovlps} is > {max_overlaps}. Halving collar to {collar_}") 
                seqs, constraints, num_ovlps, sym_map = self.get_seqs_and_constraints(
                    c, o, l, spk, spk2int,
                    collar=collar_, allow_self_overlap=allow_self_overlap,
                )
            
            fsa = self.build_topo_sort_fsa(seqs, constraints, aux_labels=True)
            
            if debug:
                sym_str = ""
                for sym, idx in fsa.symbols.items():
                    sym_str += f"{sym} {idx}\n"

                # Create k2 SymbolTable from string
                fsa.labels_sym = k2.SymbolTable.from_str(sym_str)
                fsa.draw("test_fsa.svg")
            ctc_topo = self.build_ctc_topo(
                [
                    i + spk2int[s]*(self.sp.vocab_size()-1)
                    for s, t in zip(spk, c) for i in self.sp.encode(t)
                ],
                original=original_topo
            )
            ctc_topo = ctc_topo.to(self.device)
            fsa = fsa.to(self.device)
            fsa_with_self_loop = k2.remove_epsilon_and_add_self_loops(fsa)
            fsa_with_self_loop = k2.connect(fsa_with_self_loop)
            fsa_with_self_loop = k2.arc_sort(fsa_with_self_loop)
            graph = k2.compose(
                ctc_topo,
                fsa_with_self_loop,
                treat_epsilons_specially=False,
            )
            graphs.append(graph)
        training_graphs = k2.create_fsa_vec(graphs)
        return training_graphs

    def compile_nospk(
        self,
        cuts: List[List[str]],
        offsets,
        lens,
        collar: int = 64000,
        max_overlaps: int = 6000, 
        original_topo: bool = False,
        debug: bool = False,
    ) -> k2.Fsa:
        """
            Compile a batch of transcripts into CTC-constrained decoding graphs.

            This function encodes each transcript into subword/token IDs, constructs a
            constrained alignment FSA based on per-token constraints, and composes it
            with a CTC topology graph to produce a training graph. The result is a
            `k2.FsaVec` of decoding graphs suitable for CTC training.

            :param cuts: A batch of transcripts, where each transcript is a list of string tokens.
            :type cuts: List[List[str]]
            :param offsets: A batch of starting offsets for each token in each transcript.
            :type offsets: List[List[int]] or compatible structure
            :param lens: A batch of lengths (e.g., in frames or tokens) for each token in each transcript.
            :type lens: List[List[int]] or compatible structure
            :param debug: Flag for debugging outputs
            :type debug: bool
            :param max_overlaps: the maximum number of overlapping tokens allowed
            :type max_overlaps: int
            :param original_topo: Which ctc topology to use
            :type original_topo: bool
            :return: A batched FSA (`FsaVec`) representing all input transcripts composed with a CTC topology.
            :rtype: k2.Fsa
        """
        graphs = []
        for c, o, l in zip(cuts, offsets, lens):
            # Estimate
            num_ovlps = max_overlaps + 1
            collar_ = collar
            seqs, constraints, num_ovlps, sym_map = self.get_seqs_and_constraints_no_spk(
                c, o, l, collar=collar_, allow_self_overlap=True,
            )
            while num_ovlps > max_overlaps: 
                collar_ = collar_ // 2
                logging.info(f"Number of overlaps {num_ovlps} is > {max_overlaps}. Halving collar to {collar_}") 
                seqs, constraints, num_ovlps, sym_map = self.get_seqs_and_constraints_no_spk(
                    c, o, l, collar=collar_
                )
            
            fsa = self.build_topo_sort_fsa(seqs, constraints, use_speaker=False, aux_labels=True)
            if debug:
                sym_str = ""
                for sym, idx in fsa.symbols.items():
                    sym_str += f"{sym} {idx}\n"

                # Create k2 SymbolTable from string
                fsa.labels_sym = k2.SymbolTable.from_str(sym_str)
                fsa.draw("test_fsa.svg")
            ctc_topo = self.build_ctc_topo(
                [
                    i for t in c for i in self.sp.encode(t)
                ],
                original=original_topo
            )
            ctc_topo = ctc_topo.to(self.device)
            fsa = fsa.to(self.device)
            try:
                fsa_with_self_loop = k2.remove_epsilon_and_add_self_loops(fsa)
            except:
                import pdb; pdb.set_trace()
                print() 
            fsa_with_self_loop = k2.connect(fsa_with_self_loop)
            fsa_with_self_loop = k2.arc_sort(fsa_with_self_loop)
            graph = k2.compose(
                ctc_topo,
                fsa_with_self_loop,
                treat_epsilons_specially=False,
            )
            graphs.append(graph)
        training_graphs = k2.create_fsa_vec(graphs)
        return training_graphs

    def compile_transcript(
        self,
        cuts: List[List[str]],
        offsets: List[List[int]],
        lens: List[List[int]],
        collar: int = 64000,
        max_overlaps: int = 6000,
        debug: bool = False
    ) -> k2.Fsa:
        """
            Compile a batch of transcripts into a vectorized FSA representation.

            Each transcript is converted into a constrained decoding graph (FSA),
            based on provided text segments (`cuts`), their start positions (`offsets`),
            and their lengths (`lens`). The result is a `k2.FsaVec` representing all
            transcripts in the batch.

            :param cuts: A batch of transcripts, where each transcript is a list of string tokens.
            :type cuts: List[List[str]]
            :param offsets: A batch of starting offsets corresponding to each token in each transcript.
            :type offsets: List[List[int]]
            :param lens: A batch of lengths (e.g., in frames or tokens) for each token in each transcript.
            :type lens: List[List[int]]
            :param debug: If True, enables debugging code or visualization. Default is False.
            :type debug: bool
            :return: A batched FSA (`FsaVec`) representing all input transcripts.
            :rtype: k2.Fsa
        """ 
        graphs = []
        for c, o, l in zip(cuts, offsets, lens):
            num_ovlps = max_overlaps + 1
            collar_ = collar
            seqs, constraints, num_ovlps, sym_map = self.get_seqs_and_constraints_no_spk(
                c, o, l, collar=collar_,
            )
            while num_ovlps > max_overlaps: 
                collar_ = collar_ // 2
                logging.info(f"Number of overlaps {num_ovlps} is > {max_overlaps}. Halving collar to {collar_}") 
                seqs, constraints, num_ovlps, sym_map = self.get_seqs_and_constraints_no_spk(
                    c, o, l, collar=collar_
                )
            
            fsa = self.build_topo_sort_fsa(seqs, constraints, use_speaker=False)
            if debug:
                sym_str = ""
                for sym, idx in fsa.symbols.items():
                    sym_str += f"{sym} {idx}\n"

                # Create k2 SymbolTable from string
                fsa.labels_sym = k2.SymbolTable.from_str(sym_str)
                fsa.draw("test_fsa.svg")
            graphs.append(fsa)
        training_graphs = k2.create_fsa_vec(graphs)
        return training_graphs

    
    def build_topo_sort_fsa(
        self,
        sequences: List[Tuple],
        extra_constraints: List[Tuple],
        use_speaker: bool = True,
        debug: bool = False,
        aux_labels: bool = False,
    ) -> k2.Fsa:
        """
            Construct a topologically sorted FSA from partially ordered sequences and additional constraints.

            This method builds an `Fsa` that encodes a partial order over symbols derived from one or more
            sequences. Each sequence defines an ordered path (e.g., a → b → c), and `extra_constraints`
            define additional pairwise precedence relations (e.g., a must come before d). The resulting
            FSA encodes all paths that respect the union of these constraints.

            The symbol mapping ensures that internal symbol IDs are translated to consistent token indices
            used elsewhere (e.g., in sentencepiece or token vocab). The output FSA is compatible with k2
            composition and decoding routines.

            :param sequences: A list of tuples, each representing a partially ordered sequence of symbols.
                              For example, `[('a', 'b', 'c')]` encodes `a → b → c`.
            :type sequences: List[Tuple]
            :param extra_constraints: A list of tuples specifying additional precedence constraints
                                      between symbols (e.g., `[('a', 'd')]` enforces that `a` comes before `d`).
            :type extra_constraints: List[Tuple]
            :param debug: If True, enables additional debug output or diagnostics.
            :type debug: bool
            :return: An FSA that encodes all sequences consistent with the partial orders and constraints.
            :rtype: k2.Fsa
        """ 
        # Step 1: Gather all unique symbols and assign integer labels
        all_symbols = set()
        for seq in sequences:
            all_symbols.update(seq)
        for u, v in extra_constraints:
            all_symbols.add(u)
            all_symbols.add(v)
        symbols = sorted(all_symbols)
        sym2id = {s: i for i, s in enumerate(symbols, 1)}  # k2 requires nonzero symbols
        id2sym = {i: s for s, i in sym2id.items()}
        # (vocabsize-1) Need to handle subtle case regarding not repeating the blank per speaker
        if use_speaker:
            id2orig_id = {
                i: s[0] + s[1]*(self.sp.vocab_size()-1)
                for i, s in id2sym.items()
            }
        else:
            id2orig_id = {i: s[0] for i, s in id2sym.items()} 

        N = len(symbols)
        sym2index = {s: i for i, s in enumerate(symbols)}
        index2sym = {i: s for s, i in sym2index.items()}

        # Step 2: Build partial order graph
        # Each element in seq follows the previous element
        # i.e., in abcd 
        # a->b
        # b->c
        # c->d
        preds = defaultdict(set)
        for seq in sequences:
            for u, v in zip(seq, seq[1:]):
                preds[v].add(u)
        # Add in the extra cross sequence constraints 
        for u, v in extra_constraints:
            preds[v].add(u)

        # Precompute predecessor bitmasks
        preds_bits = [0] * N
        for sym in symbols:
            idx = sym2index[sym]
            bitmask = 0
            for p in preds[sym]:
                bitmask |= (1 << sym2index[p])
            preds_bits[idx] = bitmask

        # Step 3: BFS over emitted sets using bitmasks
        state_map = {0: 0}  # emitted bitmask -> state id
        queue = deque([0])
        next_state_id = 1
        transitions = []
        final_states = set()
        all_emitted_mask = (1 << N) - 1

        while queue:
            emitted = queue.popleft()
            curr_state = state_map[emitted]

            # Check if final state
            if emitted == all_emitted_mask:
                final_states.add(curr_state)
                continue

            # Find candidate symbols to emit next
            for i in range(N):
                # Skip if already emitted
                if emitted & (1 << i):
                    continue
                # Check if all preds emitted
                if (emitted & preds_bits[i]) == preds_bits[i]:
                    new_emitted = emitted | (1 << i)
                    if new_emitted not in state_map:
                        state_map[new_emitted] = next_state_id
                        next_state_id += 1
                        queue.append(new_emitted)
                    next_state = state_map[new_emitted]
                    # The + 1 is because k2 reserves a meaning for 0
                    aux = index2sym[i][3] + 1 if use_speaker else index2sym[i][2] + 1
                    sym_id = sym2id[index2sym[i]]
                    # transitions: (src_state, dest_state, label, aux_label, score)
                    if aux_labels:
                        transitions.append((curr_state, next_state, sym_id, aux, 0.0))
                    else:
                        transitions.append((curr_state, next_state, sym_id, 0.0))

        # Step 4: Build k2 FSA from transitions and final states
        # k2 FSA text format: "src dest label [aux_label] [weight]"
        lines = []
        if aux_labels:
            for (src, dest, label, aux, weight) in transitions:
                new_label = id2orig_id[label]
                lines.append(f"{src} {dest} {new_label} {aux} {weight}")
            final_state = next_state_id 
            for s in final_states: 
                lines.append(f"{s} {final_state} -1 -1 0.0")
        else:
            for (src, dest, label, weight) in transitions:
                new_label = id2orig_id[label]
                lines.append(f"{src} {dest} {new_label} {weight}")
            final_state = next_state_id 
            for s in final_states: 
                lines.append(f"{s} {final_state} -1 0.0")
        
        lines.append(f"{final_state}")

        fsa_str = "\n".join(lines)
        fsa = k2.Fsa.from_str(fsa_str, acceptor=(not aux_labels))
        
        if debug:
            # Relabel all of the labels according to the original bpe units
            #new_labels = torch.zeros(len(fsa.arcs.values()), dtype=torch.int32)
            for i, a in enumerate(fsa.arcs.values()):
                if a[2] != -1:
                    new_labels[i] = id2orig_id[a[2].item()]
            #fsa.symbols = origsym2id
        return fsa


    

def main(args):
    
    # Your code here
    #test_make_transcript_fsa()
    #test_compose_speaker_transcripts(args)
    #test_make_multispeaker_supervision_graph(args)
    import time
    start = time.time()
    test_compile(args)
    print(f"Elapsed: {time.time() - start}s")

def test_make_transcript_fst(args):
    compiler = MDCTCGraphCompiler(args.langdir)
    fsa = compiler.make_transcript_fst("How are things going", 0)
    return fsa


def test_compose_speaker_transcripts(args):
    compiler = MDCTCGraphCompiler(args.langdir)
    
    transcript1 = "How are things"
    transcript2 = "My name is Matthew"
    fsa = compiler.compose_speaker_transcripts(
        "How are things",
        "My name is Matthew"
    )
     

def test_make_multispeaker_supervision_graph(args):
    compiler = MDCTCGraphCompiler(args.langdir)
    transcripts = [
        "How are things",
        "My name is Matthew",
        "your name is Mathias",
        "How are you"
    ]
    
    fsa_curr = compiler.make_multispeaker_supervision_graph(transcripts)
    return fsa_curr

def test_compile(args):
    compiler = MDCTCGraphCompiler(args.langdir, device="cpu")
    #transcripts = [
    #    "How are things",
    #    "My name is Matthew",
    #]   

    transcripts = [
        ["HOW ARE THINGS", "WHAT ARE YOU UP TO"],
        ["I HAVE NEVER REALLY UNDERSTOOD ALL OF WHAT MR COLTER TRIED TO TEACH ME", "ME NEITHER", "NOR HAVE I UNDERSTOOD HIS GREAT WISDOM"],
        ["SOMETIMES I FEEL BAD", "YEAH"],
    ]

    graph = compiler.compile(
        transcripts,
        [[0, 150], [0, 800, 1000], [0, 50]],
        [[150, 150], [900, 200, 500], [200, 50]],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The MDCTC Compiler")
    # parser.add_argument('--example', help='An example argument')
    parser.add_argument(
        "--langdir",
        help="The path to the directory containing the bpe.model file created"
             " by the sentencepiece module.",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    main(args)

