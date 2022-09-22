import re

def sortFunc(name: str):
    idx = re.search(r'(\d*).png', str(name)).group(1)
    return int(idx)


def sample2idx(mapdict, samplelist):
    idxlist = []
    for sample in samplelist:
        idx = mapdict[sample]
        idxlist.append(idx)
    return tuple(idxlist)