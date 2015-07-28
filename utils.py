import gzip

def OpenFile(filename, mode="r"):
    if filename.endswith(".gz"):
        return gzip.open(filename, mode)
    return open(filename, mode)
