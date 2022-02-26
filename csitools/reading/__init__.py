from tokenize import String
import csiread
import numpy as np

def extractCSI(fpath: str) -> np.ndarray:
  data = csiread.Atheros(fpath, nrxnum=2, ntxnum=5, tones=56, if_report=False)
  data.read(endian='big')
  payload_len = np.bincount(data.payload_len).argmax()
  csi = data.csi[(data.payload_len == payload_len) & (data.nc == 2)][:, :, :2, :2]
  return csi



# Также функции множественного считывания файлов по корневому пути