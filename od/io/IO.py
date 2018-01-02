# -*- coding: utf-8 -*-

import os
import codecs
from collections import Counter, defaultdict
from itertools import chain, count

import torch
import torchtext.data
import torchtext.vocab

PAD_WORD = '<blank>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'

def get_fields(data_type):
    """
    Args:
        data_type: type of the source input. Options are [text|img|audio].
    Returns:
        A dictionary whose keys are strings and whose values are the
        corresponding Field objects.
    """
    fields = {}
    if data_type == 'text':
        fields["text_context"] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)

    elif data_type == 'visual':
        def make_img(data, _):
            c = data[0].size(0)
            h = max([t.size(1) for t in data])
            w = max([t.size(2) for t in data])
            imgs = torch.zeros(len(data), c, h, w)
            for i, img in enumerate(data):
                imgs[i, :, 0:img.size(1), 0:img.size(2)] = img
            return imgs
        fields["visual_context"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=make_img, sequential=False)
        fields["text_context"] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)

    elif data_type == 'audio':
        def make_audio(data, _):
            nfft = data[0].size(0)
            t = max([t.size(1) for t in data])
            sounds = torch.zeros(len(data), 1, nfft, t)
            for i, spect in enumerate(data):
                sounds[i, :, :, 0:spect.size(1)] = spect
            return sounds

        fields["audio_context"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=make_audio, sequential=False)
        fields["text_context"] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)

    fields["response"] = torchtext.data.Field(
        init_token=BOS_WORD, eos_token=EOS_WORD,
        pad_token=PAD_WORD)

    def make_text_context(data, _):
        src_size = max([t.size(0) for t in data])
        src_vocab_size = max([t.max() for t in data]) + 1
        alignment = torch.zeros(src_size, len(data), src_vocab_size)
        for i, sent in enumerate(data):
            for j, t in enumerate(sent):
                alignment[j, i, t] = 1
        return alignment

    fields["text_context_map"] = torchtext.data.Field(
        use_vocab=False, tensor_type=torch.FloatTensor,
        postprocessing=make_text_context, sequential=False)

    def make_response(data, _):
        tgt_size = max([t.size(0) for t in data])
        alignment = torch.zeros(tgt_size, len(data)).long()
        for i, sent in enumerate(data):
            alignment[:sent.size(0), i] = sent
        return alignment

    fields["alignment"] = torchtext.data.Field(
        use_vocab=False, tensor_type=torch.LongTensor,
        postprocessing=make_response, sequential=False)

    fields["indices"] = torchtext.data.Field(
        use_vocab=False, tensor_type=torch.LongTensor,
        sequential=False)

    return fields

