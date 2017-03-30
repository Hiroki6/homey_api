#!/usr/bin python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
                        optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from models.attention import Attention
import codecs
from vocabulary.vocabulary import Vocabulary
from utils import parser
import sys

def get_response(utterance, model_filepath):

    # Settings
    dim_embed = 100

    # Train data
    # sns
    input_path = 'data/sns/utterance_wakati.txt'
    output_path = 'data/sns/response_wakati.txt'
    Vocabulary(input_path, output_path)
    
    input_path2 = 'data/broken_dialog/utterance_wakati.txt'
    output_path2 = 'data/broken_dialog/response_wakati.txt'
    Vocabulary(input_path2, output_path2)

    utterance_wakati = parser.split_into_words(utterance)

    # Model definition
    model = Attention(input_path, output_path, dim_embed)
    serializers.load_npz(model_filepath, model)
    
    response = model.predict(utterance_wakati)
    return response.replace("<eos>", "")

if __name__ == "__main__":
    argvs = sys.argv
    utterance = argvs[1]

    print get_response(utterance, "attention-models/attention-19.model")
