#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import cPickle
import math
from utils import OpenFile

class LangModel(object):
    def __init__(self, freq_deep):
        self.FreqDeep = freq_deep
        self.Frequencies = {depth: {} for depth in xrange(1, self.FreqDeep + 1)}

    @classmethod
    def LoadFromFile(cls, filename):
        with OpenFile(filename, "r") as data_reader:
            obj = cPickle.load(data_reader)
            return obj

    def SaveToFile(self, filename):
        with OpenFile(filename, "w") as data_writer:
            cPickle.dump(self, data_writer, protocol=2)

    def AddWord(self, word):
        word = word.replace(u' ', '')
        assert word
        for pos in xrange(-self.FreqDeep, len(word)):
            for depth in xrange(1, self.FreqDeep + 1):
                if pos + depth >= 0:
                    substr = word[max(pos, 0): pos + depth]
                    if len(substr) < depth:
                        if pos < 0:
                            substr = '^' * (depth - len(substr)) + substr
                        else:
                            substr = substr + '$' * (depth - len(substr))
                    self.Frequencies[depth][substr] = self.Frequencies[depth].get(substr, 0) + 1

    def OptimizeModel(self):
        cnt1 = sum(self.Frequencies[1].itervalues())
        for depth in xrange(self.FreqDeep, 0, -1):
            for k, v in self.Frequencies[depth].items():
                if depth > 1:
                    self.Frequencies[depth][k] = math.log(self.Frequencies[depth][k] * 1.0 / self.Frequencies[depth - 1][k[:-1]])
                else:
                    self.Frequencies[depth][k] = math.log(self.Frequencies[depth][k] * 1.0 / cnt1)

    def MostFreqGramms(self, depth):
        grams = sorted(self.Frequencies[depth], key=lambda w: self.Frequencies[depth][w], reverse=True)
        return u" ".join(grams[:10])

    def GetLogProbability(self, word):
        word = word.replace(u' ', '')
        p = 0
        prevPrefix = '^' * self.FreqDeep
        for ch in word:
            prefix = (prevPrefix + ch)
            if len(prefix) > self.FreqDeep:
                prefix = prefix[1:]
            while prefix and prefix not in self.Frequencies[len(prefix)]:
                prefix = prefix[1:]
            if prefix:
                p += self.Frequencies[len(prefix)][prefix]
            prevPrefix = prefix
        return p / len(word)

