import csiread
from csiread import utils
from time import time
from matplotlib import pyplot as plt

print('hi')
# Atheros CSI Tool
csifile = "./csi_data/homelocation/r_oomk_itchent_oilet/train/room.dat"
csidata = csiread.Atheros(csifile, nrxnum=2, ntxnum=5, tones=56, if_report=False)

start = time()
for i in range(1):
  csidata.read(endian='big')

print('time:', time() - start)

print(csidata.csi.shape)
print(csidata.csi[0])
print('hello')

csi = csidata.csi[:10]
phase0 = utils.np.unwrap(utils.np.angle(csi), axis=1)
phase = utils.calib(phase0, k=utils.scidx(20, 1), axis=1)
print(phase[0][:, 1, 1])
plt.plot(phase0[0][:, 1, 1])
plt.plot(phase[0][:, 1, 1])
plt.grid()
plt.show()
