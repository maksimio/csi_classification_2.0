from metawifi import LogReader
from time import time
path = './csi_data/homelocation/r_oomk_itchent_oilet/train/kitchen.dat'

# wd = WifiDf(path).set_type('abs')#.view(25)
# print(wd.df_csi_abs.shape)

start = time()
for i in range(1):
  r = LogReader(path).read()

print('time:', time() - start)
print(len(r.raw))