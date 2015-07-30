import gzip

def OpenFile(filename, mode="r"):
    if filename.endswith(".gz"):
        return gzip.open(filename, mode)
    return open(filename, mode)


class FMeasureCalculator(object):
    def __init__(self):
        self.fp, self.tp, self.fn, self.tn = 0, 0, 0, 0

    def AddResult(self, classification, expected):
        assert isinstance(classification, bool) and isinstance(expected, bool)
        if classification:
            if expected: 
                self.tp += 1
            else:
                self.fp += 1
        else:
            if expected:
                self.fn += 1
            else:
                self.tn += 1

    def GetPrecision(self):
        return self.tp * 1. / ((self.tp + self.fp) or 1)

    def GetRecall(self):
        return self.tp * 1. / ((self.tp + self.fn) or 1)

    def GetFMeasure(self):
        prec = self.GetPrecision()
        recall = self.GetRecall()
        return 2. * prec * recall / ((prec + recall) or 1)


