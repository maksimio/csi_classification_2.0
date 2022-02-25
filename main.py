import csiread
import numpy as np
import csitools

# path = './csi_data/use_in_paper/2_objects/test/D=2020-05-20_T=18-02-50--air.dat'
path = './csi_data/homelocation/r_oomk_itchent_oilet/train/room.dat'
data = csiread.Atheros(path, nrxnum=2, ntxnum=5, tones=56, if_report=False)
data.read(endian='big')

csi = data.csi[(data.payload_len == 556) & (data.nc == 2)][:, :, :2, :2]
csi = np.reshape(csi, (csi.shape[0], csi.shape[1], -1))
print(csi.shape)
ampl = np.abs(csi)
print(ampl[0])

ampl = np.transpose(ampl, (0, 2, 1))
print(ampl[0])