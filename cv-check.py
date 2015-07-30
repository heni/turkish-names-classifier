#!/usr/bin/env python2.7
#-*- coding: utf-8 -*-
import argparse
import json
import math
import random

from classifiers import IGenderClassifier, LangModelsGenderClassifier, ConstGenderClassifier, RandomGenderClassifier
from langmodel import LangModel
from utils import OpenFile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", choices=("langmodel", "man", "woman", "random"), default="langmodel")
    parser.add_argument("--depth", dest="depth", type=int, default=3, help="depth for trained language models")
    parser.add_argument("--input-data", dest="input_data", default="data/turk-names.js.gz", help="json file with source data")
    return parser.parse_args()


def Split(data, test_part, seed):
    rndGen = random.Random(seed)
    trainSet, testSet = [], []
    for nmInfo in data:
        if rndGen.random() < test_part:
            testSet.append(nmInfo)
        else:
            trainSet.append(nmInfo)
    return trainSet, testSet


class LangModelClassifierHelpers(object):
    @staticmethod
    def Train(trainSet, opts):
        mnModel, wmModel = LangModel(opts.depth), LangModel(opts.depth)
        for nmInfo in trainSet:
            name = nmInfo["name"]
            genders = set(nmInfo["gender"])
            if u'Er' in genders:
                mnModel.AddWord(name)
            if u'Ka' in genders:
                wmModel.AddWord(name)
        mnModel.OptimizeModel()
        wmModel.OptimizeModel()
        return LangModelsGenderClassifier(mnModel, wmModel)


class SimpleClassifierHelpers(object):
    @staticmethod
    def Train(trainSet, opts):
        if opts.model == "random":
            return RandomGenderClassifier()
        if opts.model == "man":
            return ConstGenderClassifier(IGenderClassifier.MAN_MARKER)
        if opts.model == "woman":
            return ConstGenderClassifier(IGenderClassifier.WOMAN_MARKER)
        raise AttributeError("wrong parameters for ConstGenderClassifier creation")


def CalcErrRate(model, data):
    goodPredictions, allPredictions = 0, 0
    for item in data:
        if u'Er' in item["gender"] or u'Ka' in item["gender"]:
            tp = model.Classify(item["name"])
            if tp in item["gender"]:
                goodPredictions += 1
            allPredictions += 1
    print goodPredictions, allPredictions, goodPredictions * 1. / allPredictions, len(data)
    return 1 - goodPredictions * 1. / allPredictions


def GetMeanAndVariance(data):
    M, S = data[0], 0
    for i in xrange(1, len(data)):
        _M = M
        M += (data[i] - M) / (i + 1)
        S += (data[i] - M) * (data[i] - _M)
    return M, S / ((len(data) - 1) or 1)


if __name__ == "__main__":
    opts = parse_args()
    allData = json.load(OpenFile(opts.input_data))
    trainFn = LangModelClassifierHelpers.Train if opts.model == "langmodel" else SimpleClassifierHelpers.Train
    errRates = []
    for _ in xrange(100):
        learn, test = Split(allData, 0.1, random.randint(1, 2**64))
        model = trainFn(learn, opts)
        errRates.append(CalcErrRate(model, test))
    eM, eS = GetMeanAndVariance(errRates)
    print "ERR-RATES: {0:.4f}±{1:.4f}".format(eM, 3 * math.sqrt(eS/len(errRates)))
