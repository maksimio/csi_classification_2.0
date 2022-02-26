import csiread
from csiread import utils
from time import time, sleep
from matplotlib import pyplot as plt

filepathes = [
  './csi_data/homelocation/r_oomk_itchent_oilet/train/kitchen.dat',
  './csi_data/homelocation/r_oomk_itchent_oilet/train/room.dat',
  './csi_data/homelocation/r_oomk_itchent_oilet/train/toilet.dat',
  './csi_data/homelocation/r_oomk_itchent_oilet/test/kitchen.dat',
  './csi_data/homelocation/r_oomk_itchent_oilet/test/room.dat',
  './csi_data/homelocation/r_oomk_itchent_oilet/test/toilet.dat',
]

csifile = "./csi_data/homelocation/r_oomk_itchent_oilet/train/room.dat"
csidata = []
start = time()
for path in filepathes:
  data = csiread.Atheros(csifile, nrxnum=2, ntxnum=5, tones=56, if_report=False)
  data.read(endian='big')
  csidata.append(data)




print(time() - start)

print(csidata[0].csi.shape)
csi = csidata[0].csi[:100]
phase0 = utils.np.unwrap(utils.np.angle(csi), axis=1)
phase = utils.calib(phase0, k=utils.scidx(20, 1), axis=1)
phaseTest = utils.np.angle(csi)

for i in [9, 18, 29]:
  plt.plot(phase0[i][:, 1, 1])
  plt.plot(phase[i][:, 1, 1])
  plt.plot(phaseTest[i][:, 1, 1])
  plt.grid()
  print(i, phase0[i][:, 1, 1], phaseTest[i][:, 1, 1])
  plt.show()
