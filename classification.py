from metawifi import WifiDf, WifiLearn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

path = './csi/homelocation/r_oomk_itchent_oilet'

wd = WifiDf(path).set_type('abs')#.view(25)
print(wd.df_csi_abs.shape)

wl = WifiLearn(*wd.prep_csi()).fit_classic().print()
exit()

wl = WifiLearn(*wd.prep_featurespace()).fit_classic().print()
wd.set_type('phase').unjump()
wl = WifiLearn(*wd.prep_csi()).fit_classic().print()
wl = WifiLearn(*wd.prep_featurespace()).fit_classic().print()
# wl = WifiLearn(*wd.prep_block_featurespace()).fit_classic().print()

# я не пробовал scal  er для пространств признаков
# я не пробовал обучать без 3 стороны (уже попробовал)
# мне нужно переснять данные более строго с фиксацией собственного положения, а также спустив роутеры на свою высоту (они под потолком)