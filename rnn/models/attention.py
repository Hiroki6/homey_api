#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
                        optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from vocabulary.vocabulary import Vocabulary

xp = np

class Attention(Vocabulary, Chain):
    def __init__(self, input_filepath, output_filepath, k):
        #BaseModel.__init__(self, input_filepath, output_filepath)
        Vocabulary.__init__(self, input_filepath, output_filepath)
        Chain.__init__(self,
                embedx = L.EmbedID(self.vocab_count, k),
                embedy = L.EmbedID(self.vocab_count, k),
                H = L.LSTM(k, k),
                Wc1 = L.Linear(k, k),
                Wc2 = L.Linear(k, k),
                W = L.Linear(k, self.vocab_count),
        )
        self.demb = k

    def __call__(self, jline, eline):
        gh = []
        for i in range(len(jline)):
            wid = self.wd2id[jline[i]]
            x_k = self.embedx(Variable(xp.array([wid], dtype=xp.int32)))
            h = self.H(x_k)
            gh.append(h.data[0])
        x_k = self.embedx(Variable(xp.array([self.wd2id['<eos>']], dtype=xp.int32)))
        tx = Variable(xp.array([self.wd2id[eline[0]]], dtype=xp.int32))
        h = self.H(x_k)
        ct = self._mk_ct(gh, h.data[0])
        h2 = F.tanh(self.Wc1(ct) + self.Wc2(h))
        accum_loss = F.softmax_cross_entropy(self.W(h2), tx)
        for i in range(len(eline)):
            wid = self.wd2id[eline[i]]
            x_k = self.embedy(Variable(xp.array([wid], dtype=xp.int32)))
            next_wid = self.wd2id['<eos>'] if (i == len(eline) - 1) else self.wd2id[eline[i+1]]
            tx = Variable(xp.array([next_wid], dtype=xp.int32))
            h = self.H(x_k)
            ct = self._mk_ct(gh, h.data)
            h2 = F.tanh(self.Wc1(ct) + self.Wc2(h))
            loss = F.softmax_cross_entropy(self.W(h2), tx)
            accum_loss += loss
        return accum_loss

    def reset_state(self):
        self.H.reset_state()

    def _mk_ct(self, gh, ht, volatile='off'):
        alp = []
        s = 0.0
        for i in range(len(gh)):
            s += xp.exp(ht.dot(gh[i]))
        ct = xp.zeros(self.demb)
        for i in range(len(gh)):
            alpi = xp.exp(ht.dot(gh[i]))/s
            ct += alpi * gh[i]

        ct = Variable(xp.array([ct], dtype=xp.float32), volatile=volatile)
        return ct


    def predict(self, jline):
        
        response = ""
        try:
            gh = []
            for i in range(len(jline)):
                wid = self.wd2id[jline[i]]
                x_k = self.embedx(Variable(xp.array([wid], dtype=xp.int32), volatile='off'))
                h = self.H(x_k)
                gh.append(xp.copy(h.data[0]))
            x_k = self.embedx(Variable(xp.array([self.wd2id['<eos>']], dtype=xp.int32), volatile='off'))
            h = self.H(x_k)
            ct = self._mk_ct(gh, h.data[0], volatile='off')
            h2 = F.tanh(self.Wc1(ct) + self.Wc2(h))
            wid = xp.argmax(F.softmax(self.W(h2)).data[0]).tolist()
            response += self.id2wd[str(wid)]
            loop = 0
            while (wid != self.wd2id['<eos>']) and (loop <= 15):
                x_k = self.embedy(Variable(xp.array([wid], dtype=xp.int32), volatile='off'))
                h = self.H(x_k)
                ct = self._mk_ct(gh, h.data, volatile='off')
                h2 = F.tanh(self.Wc1(ct) + self.Wc2(h))
                wid = xp.argmax(F.softmax(self.W(h2)).data[0]).tolist()
                response += self.id2wd[str(wid)]
                loop += 1
            response += "\n"

        except KeyError:
            return
        
        return response
