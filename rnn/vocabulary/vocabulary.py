# -*- coding:utf-8 -*-

import redis

class Vocabulary(object):
    """
    """
    r = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)
    def __init__(self, input_filepath, output_filepath):
        #r = redis.Redis(host='127.0.0.1', port=6379, db=0)

        self.wd2id = self.getWd2id()
        self.id2wd = self.getId2wd()

        # Add EOS
        if '<eos>' not in self.wd2id.keys():
            id = len(self.wd2id)
            self.wd2id['<eos>'] = id
            self.id2wd[id] = '<eos>'
            self.vocab_count = len(self.wd2id)

        else:
            self.vocab_count = len(self.wd2id)

        # Set vocabulary to dict
        self.setVocab(input_filepath)
        self.setVocab(output_filepath)

        # Set vocabulary to redis
        Vocabulary.r.hmset('wd2id', self.wd2id)
        Vocabulary.r.hmset('id2wd', self.id2wd)


    def setVocab(self, filepath):
        with open(filepath) as flines:
            for line in flines:
                lt = line.split()
                for word in lt:
                    word = unicode(word, 'utf-8')
                    if word not in self.wd2id:
                        id = len(self.wd2id)
                        self.wd2id[word] = id
                        self.id2wd[id] = word
                        self.vocab_count += 1

    def getWd2id(self):
        return Vocabulary.r.hgetall('wd2id')

    def getId2wd(self):
        return Vocabulary.r.hgetall('id2wd')


if __name__ == "__main__":
    r = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)
    wd2id = r.hgetall('wd2id')
    id2wd = r.hgetall('id2wd')

    for i in wd2id.keys():
        assert(i == id2wd[str(wd2id[i])])
