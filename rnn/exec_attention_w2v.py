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
from models.attention_w2v import Attention
import codecs
from utils import parser
import sys

def get_response(utterance, model_filepath):

    utterance_wakati = parser.split_into_words(utterance)

    # Model definition
    model = Attention()
    serializers.load_npz(model_filepath, model)

    response = model.predict(utterance_wakati)
    return response

if __name__ == "__main__":
    argvs = sys.argv
    utterance = argvs[1]

    print get_response(utterance, "attention-w2v-models/attention-2.model")
