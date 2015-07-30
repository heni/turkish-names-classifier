#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import re
import icu
import json
import collections

from utils import OpenFile

def LoadData(filename):
    TurLocale = icu.Locale('tr')
    reFormat = re.compile(u"^([^:]+):\s*(?:\(([^()]+\.)\))?\s*([^-1()]*)\s*(?:-|—|–|(?=1\.))(.*)$", re.U)
    reUnisex = re.compile(u"Erkek ve(ya)? (kadın|kız) (adı|ismi) olarak kullan(ılır|ılabilir)", re.U)
    reOptsDelim = re.compile("[.\s]")
    with open(filename) as data_reader:
        for ln in data_reader:
            ln = ln.strip().decode("utf-8")
            if ln:
                sRes = reFormat.match(ln)
                name, nameTp, nameOpts, desc = sRes.groups()
                name = unicode(icu.UnicodeString(name).toLower(TurLocale)).strip()
                nameOpts = set(opt.strip() for opt in reOptsDelim.split(nameOpts) if opt.strip())
                if reUnisex.search(desc):
                    nameOpts.add(u'Er')
                    nameOpts.add(u'Ka')
                yield collections.OrderedDict([
                    ("name", name),
                    ("type", nameTp),
                    ("gender", list(nameOpts)),
                    ("description", desc),
                ])


if __name__ == "__main__":
    data = list(LoadData("data/turk-names.txt"))
    with OpenFile("data/turk-names.js.gz", "w") as names_prn:
        names_prn.write(json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))

