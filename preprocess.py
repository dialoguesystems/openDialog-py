#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import os
import glob
import sys

from collections import Counter, defaultdict
from itertools import chain, count

import torch
import torchtext

import od
from od.io.TextDialogDataset import TextDialogDataset
import opts

def parse_args():
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.preprocess_opts(parser)
    opt = parser.parse_args()

    torch.manual_seed(opt.seed)
    return opt

def build_save_dataset(corpus_type, fields, opt, save=True):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        text_context_corpus = opt.train_src
        response_corpus = opt.train_tgt
    else:
        text_context_corpus = opt.valid_src
        response_corpus = opt.valid_tgt

    # Currently we only do preprocess sharding for corpus: data_type=='text'.
    if opt.data_type == 'text':
        """
        Process the text corpus into example_dict iterator.
        """
        text_context_iter = od.io._read_text_file(text_context_corpus, opt.src_vocab_size, "text_context")
        response_iter = od.io._read_text_file(response_corpus, opt.src_vocab_size, "response")

        dataset = TextDialogDataset(fields, text_context_iter, response_iter,
                                    src_seq_length=opt.src_seq_length,
                                    tgt_seq_length=opt.tgt_seq_length,
                                    dynamic_dict=opt.dynamic_dict,
                                    use_filter_pred=False)
        dataset.fields = []
        if save:
            torch.save(dataset, opt.save_data + '.' + corpus_type + '.pt')

    else:
        #TODO: visual & audio dialog dataset
        dataset = NotImplemented

    return dataset

def build_save_vocab(train_dataset, fields, opt, save=True):
    # We've empty'ed each dataset's `fields` attribute
    # when saving datasets, so restore them.
    for train in train_dataset:
        train.fields = fields

    fields["response"].build_vocab(*train_dataset, max_size=opt.tgt_vocab_size,
                              min_freq=opt.tgt_words_min_frequency)

    if opt.data_type == "text":
        fields["text_context"].build_vocab(*train_dataset, max_size=opt.src_vocab_size,
                                  min_freq=opt.src_words_min_frequency)

        # Merge the context and response vocabularies.
        if opt.share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            merged = sum([fields["text_context"].vocab.freqs, fields["response"].vocab.freqs], Counter())
            merged_vocab = torchtext.vocab.Vocab(merged,
                                                 specials=[od.io.IO.PAD_WORD, od.io.IO.BOS_WORD, od.io.IO.EOS_WORD],
                                                 max_size=opt.vocab_size)
            fields["text_context"].vocab = merged_vocab
            fields["response"].vocab = merged_vocab

        if save:
            torch.save(fields, opt.save_data + '.fields.pt')


def main():
    opt = parse_args()

    print('Preparing for training ...')
    fields = od.io.get_fields(opt.data_type)

    print("Building & saving training data...")
    train_datasets = build_save_dataset('train', fields, opt)

    print("Building & saving vocabulary...")
    build_save_vocab(train_datasets, fields, opt)

    print("Building & saving validation data...")
    build_save_dataset('valid', fields, opt)

if __name__ == "__main__":
    main()