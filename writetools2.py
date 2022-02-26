from os import listdir
from csitools import reading
import csitools.processing as csip
import csitools.reading as csir
import csitools.viewing as csiv
import csitools.models as csim
import numpy as np

categories = ['air', 'bottle']
dirpath = './csi_data/use_in_paper/2_objects/train'
frameHeight = 50

for fname in listdir(dirpath):
  csi = csir.extractCSI(dirpath + '/' + fname)
  csi = csip.extractAm(csi)
  csi = csip.reshape4x56(csi)
  # csi = np.diff(csi)

  csi = np.expand_dims(csi, axis=0)
  print(csi.shape)
  csi = csi[:,:csi.shape[1]//frameHeight*frameHeight,:,:]
  print(csi.shape)
  csi = np.reshape(csi, (-1, frameHeight, 4, csi.shape[3]))
  print(csi.shape)
  csi = np.transpose(csi, (0,2,1,3))
  print(csi.shape)
  break
  # csi = csip.down(csi)
  print(csi.shape, csir.match(categories, fname))
