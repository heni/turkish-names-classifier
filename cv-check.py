#!/usr/bin/env python2.7
#-*- coding: utf-8 -*-
import argparse
import json
import math
import random

from classifiers import IGenderClassifier, LangModelsGenderClassifier, ConstGenderClassifier, RandomGenderClassifier
from langmodel import LangModel
from utils import FMeasureCalculator, OpenFile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", choices=("langmodel", "man", "woman", "random"), default="langmodel")
    parser.add_argument("--input-data", dest="input_data", default="data/turk-names.js.gz", help="json file with source data")
    parser.add_argument("--depth", dest="depth", type=int, default=3, help="depth for trained language models")
    parser.add_argument("--random-coeff", dest="random_coeff", type=float, default=.5, help="threshold for random classifier")
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
            if IGenderClassifier.MAN_MARKER in genders:
                mnModel.AddWord(name)
            if IGenderClassifier.WOMAN_MARKER in genders:
                wmModel.AddWord(name)
        mnModel.OptimizeModel()
        wmModel.OptimizeModel()
        return LangModelsGenderClassifier(mnModel, wmModel)


class SimpleClassifierHelpers(object):
    @staticmethod
    def Train(trainSet, opts):
        if opts.model == "random":
            return RandomGenderClassifier(opts.random_coeff)
        if opts.model == "man":
            return ConstGenderClassifier(IGenderClassifier.MAN_MARKER)
        if opts.model == "woman":
            return ConstGenderClassifier(IGenderClassifier.WOMAN_MARKER)
        raise AttributeError("wrong parameters for ConstGenderClassifier creation")


def CalcErrRate(model, data):
    goodPredictions, allPredictions = 0, 0
    for item in data:
        if IGenderClassifier.MAN_MARKER in item["gender"] or IGenderClassifier.WOMAN_MARKER in item["gender"]:
            tp = model.Classify(item["name"])
            if tp in item["gender"]:
                goodPredictions += 1
            allPredictions += 1
    print goodPredictions, allPredictions, goodPredictions * 1. / allPredictions, len(data)
    return 1 - goodPredictions * 1. / allPredictions


def CalcFMeasures(model, data):
    mnTracker, wmTracker = FMeasureCalculator(), FMeasureCalculator()
    for item in data:
        tp = model.Classify(item["name"])
        if tp == IGenderClassifier.MAN_MARKER:
            mnTracker.AddResult(True, IGenderClassifier.MAN_MARKER in item["gender"])
            wmTracker.AddResult(False, IGenderClassifier.WOMAN_MARKER in item["gender"])
        elif tp == IGenderClassifier.WOMAN_MARKER:
            mnTracker.AddResult(False, IGenderClassifier.MAN_MARKER in item["gender"])
            wmTracker.AddResult(True, IGenderClassifier.WOMAN_MARKER in item["gender"])
    return mnTracker.GetFMeasure(), wmTracker.GetFMeasure()


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
    mnF1List, wmF1List = [], []
    for _ in xrange(100):
        learn, test = Split(allData, 0.1, random.randint(1, 2**64))
        model = trainFn(learn, opts)
        mnF1, wmF1 = CalcFMeasures(model, test)
        mnF1List.append(mnF1)
        wmF1List.append(wmF1)
    eM, eS = GetMeanAndVariance(mnF1List)
    print "F1(man):   {0:.4f}±{1:.4f}".format(eM, 3 * math.sqrt(eS/len(mnF1List)))
    eM, eS = GetMeanAndVariance(wmF1List)
    print "F1(woman): {0:.4f}±{1:.4f}".format(eM, 3 * math.sqrt(eS/len(wmF1List)))
