import csitools.processing as csip
import csitools.reading as csir
import csitools.viewing as csiv
import numpy as np

path = './csi_data/use_in_paper/2_objects/train/D=2020-01-02_T=21-09-35--air.dat'
csi = csir.extractCSI(path)
csi = csip.reshape4x56(csi)
csi = csip.extractPh(csi)
csi = np.diff(csi)
csi = np.reshape(csi, (csi.shape[0], -1))
csi = csip.down(csi)

csiv.imsave('./newtest4.jpg', csi)