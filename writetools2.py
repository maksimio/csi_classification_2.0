from os import listdir
from csitools import reading
import csitools.processing as csip
import csitools.reading as csir
import csitools.viewing as csiv
import csitools.models as csim
import numpy as np

categories = ['air', 'bottle']
dirpath = './csi_data/use_in_paper/2_objects/train'
for fname in listdir(dirpath):
  csi = csir.extractCSI(dirpath + '/' + fname)
  csi = csip.extractAm(csi)
  csi = csip.reshape4x56(csi)
  csi = np.diff(csi)
  csi = np.reshape(csi, (csi.shape[0], -1))
  csi = csip.down(csi)
  print(csi.shape, csir.match(categories, fname))
