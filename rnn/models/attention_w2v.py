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
from gensim.models.word2vec import Word2Vec

xp = np

class Attention(Vocabulary, Chain):
    def __init__(self):
        w2v_path = 'data/word2vec/word2vec.gensim.model'
        self.word2vec = Word2Vec.load(w2v_path)
        self.embed_dim = self.word2vec.vector_size
        self.lstm_out = 300
        Chain.__init__(self,
                H1 = L.LSTM(self.embed_dim, self.lstm_out),
                H2 = L.LSTM(self.lstm_out, self.lstm_out),
                Wc1 = L.Linear(self.lstm_out, 100),
                Wc2 = L.Linear(self.lstm_out, 100),
                W = L.Linear(100, self.embed_dim),
        )

    def __call__(self, jline, eline):
        gh = []
        for i in range(len(jline)):
            #print jline[i] in self.word2vec.wv.vocab
            word_vec = self.word2vec[jline[i]] if jline[i] in self.word2vec.wv.vocab else xp.random.uniform(-0.1, 0.1, self.embed_dim)
            x = Variable(xp.array([word_vec], dtype=xp.float32))
            h1 = self.H1(x)
            h2 = self.H2(h1)
            gh.append(h2.data[0])
        x = Variable(xp.array([self.word2vec[u'EOS']], dtype=xp.float32))
        tx_ = self.word2vec[eline[0]] if eline[0] in self.word2vec.wv.vocab else xp.random.uniform(-0.1, 0.1, self.embed_dim)
        tx = Variable(xp.array([tx_], dtype=xp.float32))
        h1 = self.H1(x)
        h2 = self.H2(h1)
        ct = self._mk_ct(gh, h2.data[0])
        h3 = F.tanh(self.Wc1(ct) + self.Wc2(h2))
        accum_loss = F.mean_squared_error(self.W(h3), tx)
        for i in range(len(eline)):
            word_vec = self.word2vec[eline[i]] if eline[i] in self.word2vec.wv.vocab else xp.random.uniform(-0.1, 0.1, self.embed_dim)
            x = Variable(xp.array([word_vec], dtype=xp.float32))

            if i == len(eline) - 1:
                next_word_vec = self.word2vec[u'EOS']
            else:
                next_word_vec = self.word2vec[eline[i+1]] if eline[i+1] in self.word2vec.wv.vocab else xp.random.uniform(-0.1, 0.1, self.embed_dim)
            tx = Variable(xp.array([next_word_vec], dtype=xp.float32))
            h1 = self.H1(x)
            h2 = self.H2(h1)
            ct = self._mk_ct(gh, h2.data)
            h3 = F.tanh(self.Wc1(ct) + self.Wc2(h2))
            loss = F.mean_squared_error(self.W(h3), tx)
            accum_loss += loss
        return accum_loss

    def reset_state(self):
        self.H1.reset_state()
        self.H2.reset_state()

    def _mk_ct(self, gh, ht, volatile='off'):
        alp = []
        s = 0.0
        for i in range(len(gh)):
            s += xp.exp(ht.dot(gh[i]))
        ct = xp.zeros(self.lstm_out)
        for i in range(len(gh)):
            alpi = xp.exp(ht.dot(gh[i]))/s
            ct += alpi * gh[i]

        ct = Variable(xp.array([ct], dtype=xp.float32), volatile=volatile)
        return ct

    def predict(self, jline):
    
        response = ""
        gh = []
        for i in range(len(jline)):
            word_vec = self.word2vec[jline[i]] if jline[i] in self.word2vec.wv.vocab else xp.random.uniform(-0.1, 0.1, self.embed_dim)
            x = Variable(xp.array([word_vec], dtype=xp.float32))
            h1 = self.H1(x)
            h2 = self.H2(h1)
            gh.append(xp.copy(h1.data[0]))
        x = Variable(xp.array([self.word2vec[u'EOS']], dtype=xp.float32))
        h1 = self.H1(x)
        h2 = self.H2(h1)
        ct = self._mk_ct(gh, h2.data[0])
        h3 = F.tanh(self.Wc1(ct) + self.Wc2(h2))
        out_vec = self.W(h3)

        response += self.word2vec.wv.similar_by_vector(cuda.to_cpu(out_vec.data[0]))[0][0]
        loop = 0
        while (not np.array_equal(out_vec.data[0], self.word2vec[u'EOS'])) and (loop <= 30):
            x = Variable(xp.array([out_vec.data[0]], dtype=xp.float32))
            h1 = self.H1(x)
            h2 = self.H2(h1)
            ct = self._mk_ct(gh, h2.data)
            h3 = F.tanh(self.Wc1(ct) + self.Wc2(h2))
            out_vec = self.W(h3)
            response += self.word2vec.wv.similar_by_vector(cuda.to_cpu(out_vec.data[0]))[0][0]
            loop += 1
        response += "\n"
 
        return response
