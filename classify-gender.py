#!/usr/bin/env python2.7
import argparse
import icu
import sys

from langmodel import LangModel
from classifiers import IGenderClassifier, LangModelsGenderClassifier, ConstGenderClassifier, RandomGenderClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", choices=("langmodel", "man", "woman", "random"), default="langmodel")
    parser.set_defaults(mnmodel_filename="data/mnmodel.dat")
    parser.set_defaults(wmmodel_filename="data/wmmodel.dat")
    return parser.parse_args()


def GetClassifier(opts):
    if opts.model == "langmodel":
        return LangModelsGenderClassifier(
            LangModel.LoadFromFile(opts.mnmodel_filename),
            LangModel.LoadFromFile(opts.wmmodel_filename)
        )
    elif opts.model == "man":
        return ConstGenderClassifier(IGenderClassifier.MAN_MARKER)
    elif opts.model == "woman":
        return ConstGenderClassifier(IGenderClassifier.WOMAN_MARKER)
    elif opts.model == "random":
        return RandomGenderClassifier()
    else:
        assert False, "non-reachable code"


if __name__ == "__main__":
    TrLocale = icu.Locale("tr")
    opts = parse_args()
    model = GetClassifier(opts)
    for ln in sys.stdin.xreadlines():
        name = unicode(icu.UnicodeString(ln.strip().decode("utf-8")).toLower(TrLocale))
        prediction, aux_info = model.ClassifyAux(name)
        print u"{0} classified as {1}\t{2}".format(
            name,
            "MAN" if prediction == IGenderClassifier.MAN_MARKER else "WOMAN",
            " ".join("{0}={1}".format(k, v) for k, v in aux_info.items())
        ).encode("utf-8")
