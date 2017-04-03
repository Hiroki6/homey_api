#!/usr/bin/python
# -*- coding: utf-8 -*-

import MeCab

def split_into_words(text):
    """
    文章を単語に分割
    """
    text = text.encode("utf-8")
    mecab = MeCab.Tagger("-Owakati")
    node = mecab.parseToNode(text)
    words = []
    while node:
        parse = node.surface.decode("utf-8")
        words.append(parse)
        node = node.next
    return words[1:-1]

