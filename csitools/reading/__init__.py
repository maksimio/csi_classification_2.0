from tokenize import String
from typing import Dict, List
import csiread
import numpy as np
from os import listdir
from re import search

def extractCSI(fpath: str) -> np.ndarray:
  data = csiread.Atheros(fpath, nrxnum=2, ntxnum=5, tones=114, if_report=False)
  data.read(endian='big')
  payload_len = np.bincount(data.payload_len).argmax()
  csi = data.csi[(data.payload_len == payload_len) & (data.nc == 2)][:, :, :2, :2]
  return csi



# Также функции множественного считывания файлов по корневому пути

def extractFrom(fdir: List[str]):
  data = []
  for fname in listdir(fdir):
    data.append({'filename': fname, 'csi': extractCSI(fdir + '/' + fname)})

  return data

def matchCategories(categories: List[str], csiList: Dict):
  for item in csiList:
    for cat in categories:
      if search('.*' + cat + '.*', item['filename']):
        item['category'] = cat
        break


def match(categories: List[str], fname) -> str:
    for i, cat in enumerate(categories):
      if search('.*' + cat + '.*', fname):
        return i

    return -1