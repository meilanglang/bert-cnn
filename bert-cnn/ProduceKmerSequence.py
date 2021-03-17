#!/usr/bin/env python
# coding: utf-8

import logbook
import re
import os
from Bio import SeqIO
from attic_util import util
from itertools import islice
import numpy as np
from collections import Counter
import logbook

class SeqGenerator:
    def __init__(self,sequences,seqlen_ulim=250):
        self.sequences = sequences
        self.seqlen_ulim = seqlen_ulim
    def generator(self, rng):
        segment=self.sequences
        yield segment
        #print(segment)

class SeqFragmenter:
    """
    Split a sequence into small sequences based on some criteria, e.g. 'N' characters
    """
    def __init__(self):
        pass

    def get_acgt_seqs(self, seq):
        return remove_empty(re.split(r'[^ACGTNacgtn]+', str(seq)))

def remove_empty(str_list):
    return filter(bool, str_list)  # fastest way to remove empty string


class SlidingKmerFragmenter:
    """
    Slide only a single nucleotide
    """
    def __init__(self, k_low, k_high):
        self.k_low = k_low
        self.k_high = k_high
    
    def apply(self, rng, seq):
        kmer = [seq[i: i + rng.randint(self.k_low, self.k_high + 1)] for i in range(len(seq) - self.k_high + 1)]
        return kmer


class SeqMapper:
    def __init__(self, use_revcomp=True):
        self.use_revcomp = use_revcomp

    def apply(self,seq): 
        seq = seq.lower()
        return seq


class KmerSeqIterable:
    def __init__(self,rand_seed,seq_generator, mapper,seq_fragmenter,kmer_fragmenter):
        self.logger = logbook.Logger(self.__class__.__name__)
        self.seq_generator = seq_generator
        self.mapper = mapper
        self.kmer_fragmenter = kmer_fragmenter
        self.seq_fragmenter = seq_fragmenter
        self.rand_seed = rand_seed
        self.iter_count = 0

    def __iter__(self):
        self.iter_count += 1
        rng = np.random.RandomState(self.rand_seed)
        for seq in self.seq_generator.generator(rng):
            seq =self.mapper.apply(seq)
            acgt_seq_splits = list(self.seq_fragmenter.get_acgt_seqs(seq))
            for acgt_seq in acgt_seq_splits:
                kmer_seqs = self.kmer_fragmenter.apply(rng, acgt_seq)# list of strings 
                #print(kmer_seqs)
                yield kmer_seqs
                