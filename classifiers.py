import hashlib
import struct
from langmodel import LangModel


class IGenderClassifier(object):
    WOMAN_MARKER = "Ka"
    MAN_MARKER = "Er"

    def __init__(self):
        pass

    def Classify(self, name):
        """returns most probable gender"""
        return self.ClassifyAux(name)[0]

    def ClassifyAux(self, name):
        """return tuple with most probable gender
        and some classifier-related auxiliar information"""
        raise NotImplemented()


class LangModelsGenderClassifier(IGenderClassifier):

    def __init__(self, mnModel, wmModel):
        if not isinstance(mnModel, LangModel) or not isinstance(wmModel, LangModel):
            raise AttributeError("incorrect parameters")
        self.MnModel = mnModel
        self.WmModel = wmModel
        super(LangModelsGenderClassifier, self).__init__()

    def ClassifyAux(self, name):
        mnProb = self.MnModel.GetLogProbability(name)
        wmProb = self.WmModel.GetLogProbability(name)
        if mnProb > wmProb:
            return self.MAN_MARKER, {"mn-prob": mnProb, "wm-prob": wmProb}
        else:
            return self.WOMAN_MARKER, {"mn-prob": mnProb, "wm-prob": wmProb}


class ConstGenderClassifier(IGenderClassifier):

    def __init__(self, mark):
        if mark not in (self.WOMAN_MARKER, self.MAN_MARKER):
            raise AttributeError("incorrect marker for ConstGenderClassifier")
        self.Marker = mark
        super(ConstGenderClassifier, self).__init__()

    def ClassifyAux(self, name):
        return self.Marker, {}


class RandomGenderClassifier(IGenderClassifier):
    BASE = 2. ** 64

    def __init__(self, threshold = .5):
        self.MnTreshold = threshold
        super(RandomGenderClassifier, self).__init__()

    def ClassifyAux(self, name):
        value = struct.unpack("QQ", hashlib.md5(name.encode("utf-8")).digest())[0] / self.BASE
        if value < self.MnTreshold:
            return self.MAN_MARKER, {"value": value}
        else:
            return self.WOMAN_MARKER, {"value": value}
