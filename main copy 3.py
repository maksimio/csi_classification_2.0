import csiread
import numpy as np
import csitools
import cv2

# Чтение csi
path = './csi_data/homelocation/r_oomk_itchent_oilet/train/kitchen.dat'
data = csiread.Atheros(path, nrxnum=2, ntxnum=5, tones=56, if_report=False)
data.read(endian='big')
csi = data.csi[(data.payload_len == 556) & (data.nc == 2)][:, :, :2, :2]

ampl = np.unwrap(np.angle(csi), axis=1)
ampl = csiread.utils.calib(ampl, k=csiread.utils.scidx(20, 1), axis=1)
# ampl = np.abs(csi)


ampl = np.reshape(ampl, (ampl.shape[0], ampl.shape[1], -1))
ampl = np.transpose(ampl, (0, 2, 1))
ampl = np.diff(ampl)
ampl = np.reshape(ampl, (ampl.shape[0], -1))


print(ampl.max(), ampl.min())
ampl = ampl[:30000]
ampl = ampl - ampl.min()
print(ampl.max(), ampl.min())
ampl =  ampl / ampl.max() * 255
cv2.imwrite('./test32.jpg', np.abs(ampl))
print(ampl.shape, ampl.max(), ampl.min())