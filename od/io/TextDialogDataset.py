# -*- coding: utf-8 -*-

import os
import codecs
from collections import Counter, defaultdict
from itertools import chain, count

import torch
import torchtext.data
import torchtext.vocab

class TextDialogDataset(torchtext.data.Dataset):
    """ Dataset for data_type=="text" & just for single turn dialog system for chit-chat.

        Build Example objects, Field objects, and filter_pred function
        from text corpus.

        Args:
            fields: a dictionary of Field objects. Keys are like 'text_context',
                    'response', 'text_context_map', and 'alignment'.
            text_contexts_iter: preprocessed text_context_dict iterator.
            responses_iter: preprocessed response__dict iterator.
            src_seq_length: maximum source sequence length.
            tgt_seq_length: maximum target sequence length.
            dynamic_dict: create dynamic dictionaries?
            use_filter_pred: use a custom filter predicate to filter examples?


    """
    def __init__(self,fields, text_context_iter, response_iter,
                                    src_seq_length,
                                    tgt_seq_length,
                                    dynamic_dict,
                                    use_filter_pred=False):
        examples, fields, filter_pred = self._process_corpus(fields, text_context_iter, response_iter, src_seq_length, tgt_seq_length, dynamic_dict, use_filter_pred)
        super(TextDialogDataset, self).__init__(examples, fields, filter_pred)


    def sort_key(self, ex):
        return -len(ex.text_context)

    def _process_corpus(self, fields, src_examples_iter, tgt_examples_iter,
                        src_seq_length=0, tgt_seq_length=0,
                        dynamic_dict=True, use_filter_pred=True):
        self.data_type = 'text'

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.
        if tgt_examples_iter is not None:
            examples_iter = (_join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        if dynamic_dict:
            examples_iter = self._dynamic_dict(examples_iter)

        # Peek at the first to see which fields are used.
        ex, examples_iter = _peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)
        out_examples = (_construct_example_fromlist(ex_values, out_fields)
                        for ex_values in example_values)

        def filter_pred(example):
            return 0 < len(example.src) <= src_seq_length \
               and 0 < len(example.tgt) <= tgt_seq_length

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        return out_examples, out_fields, filter_pred

    def _dynamic_dict(self, examples_iter):
        for example in examples_iter:
            src = example["src"]
            src_vocab = torchtext.vocab.Vocab(Counter(src))
            self.src_vocabs.append(src_vocab)
            # Mapping source tokens to indices in the dynamic dict.
            src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
            example["src_map"] = src_map

            if "tgt" in example:
                tgt = example["tgt"]
                mask = torch.LongTensor(
                        [0] + [src_vocab.stoi[w] for w in tgt] + [0])
                example["alignment"] = mask
            yield example

def _construct_example_fromlist(data, fields):
    ex = torchtext.data.Example()
    for (name, field), val in zip(fields, data):
        if field is not None:
            setattr(ex, name, field.preprocess(val))
        else:
            setattr(ex, name, val)
    return ex

def _join_dicts(*args):
    """
    Args:
        dictionaries with disjoint keys.
    Returns:
        a single dictionary that has the union of these keys.
    """
    return dict(chain(*[d.items() for d in args]))

def _peek(seq):
    """
    Args:
        seq: an iterator.

    Returns:
        the first thing returned by calling next() on the iterator
        and an iterator created by re-chaining that value to the beginning
        of the iterator.
    """
    first = next(seq)
    return first, chain([first], seq)

def _read_text_file(path, truncate, side):
    """
    Args:
        path: location of a src or tgt file.
        truncate: maximum sequence length (0 for unlimited).

    Yields:
        word for each line.
    """
    assert side in ["text_context", "visual_context", "response"]

    if path is None:
        return None

    with codecs.open(path, "r", "utf-8") as corpus_file:
        for i, line in enumerate(corpus_file):
            words = line.strip().split()
            if truncate:
                words = line[:truncate]
            example_dict = {side: words, "indices": i}
            yield example_dict
