#!/usr/bin/python
# -*- coding: utf-8 -*-

import MeCab

def split_into_words(text):
    """
    文章を単語に分割
    """
    mecab = MeCab.Tagger("-Owakati")
    node = mecab.parseToNode(text)
    words = []
    while node:
        text = node.surface
        words.append(unicode(text, 'utf-8'))
        node = node.next
    return words[1:-1]

