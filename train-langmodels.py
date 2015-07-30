#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import argparse
import gzip
import json

from langmodel import LangModel
from utils import OpenFile


def TrainModels(data, depth):
    mnModel = LangModel(depth)
    wmModel = LangModel(depth)
    for nmInfo in data:
        name = nmInfo["name"]
        genders = set(nmInfo["gender"])
        if u'Er' in genders:
            mnModel.AddWord(name)
        if u'Ka' in genders:
            wmModel.AddWord(name)
    mnModel.OptimizeModel()
    wmModel.OptimizeModel()
    return mnModel, wmModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", dest="depth", type=int, default=3, help="depth for trained language models")
    parser.add_argument("--mn-model", dest="mnmodel_filename", default="data/mnmodel.dat", help="file to save lang model for man names")
    parser.add_argument("--wm-model", dest="wmmodel_filename", default="data/wmmodel.dat", help="file to save lang model for woman names")
    parser.add_argument("--input-data", dest="input_data", default="data/turk-names.js.gz", help="json file with source data")
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()
    data = json.load(OpenFile(opts.input_data))
    mnModel, wmModel = TrainModels(data, opts.depth)
    mnModel.SaveToFile(opts.mnmodel_filename)
    wmModel.SaveToFile(opts.wmmodel_filename)

