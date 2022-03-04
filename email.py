import csitools.processing as csip
import csitools.reading as csir
import csitools.viewing as csiv
import numpy as np
from matplotlib import pyplot as plt

path = './csi_data/use_in_paper/2_objects/train/D=2020-01-02_T=21-09-35--air.dat'
path = './csi_data/new2022/new40mhztest/train/bottle1.dat'
csi = csir.extractCSI(path)
csi = csip.extractAm(csi)
csi = csip.reshape4x56(csi)
# csi = np.diff(csi)
csi = np.reshape(csi, (csi.shape[0], -1))
csi = csip.down(csi)
# csiv.imsave('./phase.jpg', csi)
print(csi.shape)
plt.plot([1,2,3])
plt.savefig('/tmp/output.pdf', format='pdf')
plt.show()