# -*-coding:utf-8-*-

import MeCab
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))

"""
dict: {文字: 値}
"""
noun_dic = {} # 名詞
adjective_dic = {} # 形容詞
verb_dic = {} # 動詞
adverb_dic = {} # 副詞
with open(os.path.join(BASE, "pn_ja.dic.txt")) as f:
    for line in f:
        words = line.replace("\n", "").split(":")
        kanji = words[0].decode("shift-jis", "ignore").encode("utf-8")
        hiragana = words[1].decode("shift-jis", "ignore").encode("utf-8")
        part_of_word = words[2].decode("shift-jis", "ignore").encode("utf-8")
        value = float(words[3].decode("shift-jis", "ignore").encode("utf-8"))
        if part_of_word == "名詞":
            noun_dic[kanji] = value
            noun_dic[hiragana] = value
        elif part_of_word == "形容詞":
            adjective_dic[kanji] = value
            adjective_dic[hiragana] = value
        elif part_of_word == "動詞":
            verb_dic[kanji] = value
            verb_dic[hiragana] = value
        else:
            adverb_dic[kanji] = value
            adverb_dic[hiragana] = value

m = MeCab.Tagger("-Ochasen")

"""
[(単語, 品詞)]
"""
def parse_by_mecab(sentence):
    vocabs = m.parse(sentence).split("\n")
    results = []
    for vocab in vocabs:
        if vocab == 'EOS':
            break
        ochasens = vocab.split("\t")
        word = ochasens[2]
        part_of_word = ochasens[3].split("-")[0]
        results.append((word, part_of_word))
    
    return results

def judge_pn(sentense):
    morph_words = parse_by_mecab(sentense)
    values = 0.0
    for morph_word in morph_words:
        word = morph_word[0]
        part_of_word = morph_word[1]
        if part_of_word == "名詞":
            if noun_dic.has_key(word):
                values += noun_dic[word]
        elif part_of_word == "形容詞":
            if adjective_dic.has_key(word):
                values += adjective_dic[word]
        elif part_of_word == "動詞":
            if verb_dic.has_key(word):
                values += verb_dic[word]
        elif part_of_word == "副詞":
            if adverb_dic.has_key(word):
                values += adverb_dic[word]
        else:
            continue
    return values

if __name__ == "__main__":
    
    argvs = sys.argv

    if len(argvs) < 1:
        sys.exit(0)

    sentense = argvs[1]
    print judge_pn(sentense)
