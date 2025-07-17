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
from typing import Union, Optional, List
import torch


# For now, this only works on CPU as far as I can tell
class MDCTCGraphCompiler(object):
    def __init__(
        self,
        lang_dir: Path,
        device: Union[str, torch.device] = "cpu",
    ):
        self.device = device
        # Create a map from (speaker, token) tuples --> int
        lang_dir = Path(lang_dir)
        bpe_model_file = lang_dir / "bpe.model"
        sp = spm.SentencePieceProcessor() 
        sp.load(str(bpe_model_file))
        self.sp = sp


    def build_ctc_topo(self, symbols: List[int]) -> k2.Fsa:
        '''Builds a CTC topology over the provided symbols only.'''
        assert 0 not in symbols, "Symbol 0 is reserved for the blank token in CTC."
        
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
            arcs.append(f"{i} {i} {s1} 0 0.0")     # self-loop with symbol
            arcs.append(f"{i} 0 0 0 0.0")  # to blank
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

    def make_transcript_fsa(self, transcript):
        #labels = [int(i) for i in transcript.split()]
        labels = self.sp.encode(transcript)
        fst = k2.linear_fsa(labels)
        fst = fst.to(self.device)
        return k2.arc_sort(fst)    

    def compose_speaker_fsts(self, fst1, fst2):
        fst2 = k2.invert(fst2)
        supervision_lattice = k2.compose(fst1, fst2,
            treat_epsilons_specially=True,
        )
    
        new_fsa = supervision_lattice.clone()
    
        # Extract original labels
        labels = supervision_lattice.labels.clone()
        aux_labels = supervision_lattice.aux_labels.clone()
    
        # Define new labels:
        # - if label != EPS → use label
        # - else if aux_label != EPS → use aux_label
        # - else → use EPS
        new_labels = torch.where(
            labels != 0,
            labels,
            torch.where(
                aux_labels != 0, aux_labels,
                torch.tensor(0, dtype=torch.int32)
            )
        )
    
        # Set the new labels
        new_fsa.labels = new_labels
    
        # Remove aux_labels since it's now an FSA
        delattr(new_fsa, "aux_labels")
        return new_fsa

    def convert_fsa_to_fst_with_epsilon(self, fsa):
        if hasattr(fsa, 'aux_labels'):
            return fsa
        # Step 2: Convert to WFST by setting aux_labels to 0 (epsilon)
        fst = fsa.clone()
    
        # Get arcs
        arcs = fst.arcs.values()
    
        # Identify final state
        final_arc_mask = arcs[:, 2] == -1
        final_states = set(arcs[final_arc_mask, 1].tolist())
   
    
        # Prepare aux_labels: 0 (epsilon) by default
        aux_labels = torch.zeros(len(arcs), dtype=torch.int32)
        aux_labels = aux_labels.to(self.device)
    
        # For arcs leading to the final state, set aux_label = label
        for i, arc in enumerate(arcs):
            if arc[1].item() in final_states:
                aux_labels[i] = arc[2]  # or any other label if you prefer
        fst.aux_labels = aux_labels
    
        return fst

    def make_multispeaker_supervision_graph(self, transcripts: List[str]):
        t0 = transcripts[0]
        fsa_curr = self.make_transcript_fsa(t0)
        if len(transcripts) == 1:
            return fsa_curr.to(self.device)

        fsa_curr = fsa_curr.to(self.device)
        for t in transcripts[1:]:
            fsa = self.make_transcript_fsa(t)
            fsa = fsa.to(self.device)
            fst = self.convert_fsa_to_fst_with_epsilon(fsa)
            fst_curr = self.convert_fsa_to_fst_with_epsilon(fsa_curr)
            fst_curr = k2.arc_sort(fst_curr)
            fsa_curr = self.compose_speaker_fsts(fst_curr, fst)
        fsa_curr = k2.connect(fsa_curr)
        fsa_curr = k2.arc_sort(fsa_curr)
        return fsa_curr

    def compile(self, cuts: List[List[str]]) -> k2.Fsa:
        graphs = []
        for c in cuts:
            fst = self.make_multispeaker_supervision_graph(c)
            ctc_topo = self.build_ctc_topo([int(i) for t in c for i in self.sp.encode(t)])
            ctc_topo = ctc_topo.to(self.device)
            fsa_with_self_loop = k2.remove_epsilon_and_add_self_loops(fst)
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
        ["How are things", "What are you up to"],
        ["I have never really understood all of what mr colter tried to teach me", "me neither"],
        ["Sometimes I feel bad", "yeah"],
    ]
    graph = compiler.compile(transcripts)

#def test_add_self_loops():
#    compiler = MDCTCGraphCompiler(args.langdir)
#    fsa = compiler.make_transcript_fsa("How are things going")
#    
#    fsa = compiler.add_self_loops(fsa, 25000)
#    import pdb; pdb.set_trace()
#    return fsa


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

