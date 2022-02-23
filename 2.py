from metawifi import WifiDf, WifiLearn


# path = './csi/homelocation/three place'
path = './csi/use_in_paper/4_objects'

wd = WifiDf(path).set_type('abs')
x1, y1, x2, y2 = wd.prep_block_featurespace()
wl = WifiLearn(x1, y1, x2, y2)
res = wl.select_best(3)
print(res)