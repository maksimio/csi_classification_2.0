from metawifi import WifiDf, WifiLearn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

alpha = 0.5
path = './csi/use_in_paper/4_objects'
# path = './csi/homelocation/three place'

wd = WifiDf(path).set_type('abs')
x1, y1, x2, y2 = wd.prep_featurespace()
df = pd.DataFrame(x1.copy())
df['target'] = y1

wl = WifiLearn(x1, y1, x2, y2)
selected_df = wl.select_best(3)

features = list(selected_df.columns.astype(str))
sns.lmplot(x=features[0], y=features[1], data=df, hue='target', fit_reg=False, scatter_kws={'alpha': alpha})
plt.grid()

sns.lmplot(x=features[1], y=features[2], data=df, hue='target', fit_reg=False, scatter_kws={'alpha': alpha})
plt.grid()

sns.lmplot(x=features[2], y=features[0], data=df, hue='target', fit_reg=False, scatter_kws={'alpha': alpha})
plt.grid()
plt.show()
