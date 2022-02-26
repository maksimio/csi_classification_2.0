import csitools.processing as csip
import csitools.reading as csir

path = './csi_data/use_in_paper/2_objects/train/D=2020-01-02_T=21-09-35--air.dat'
csi = csir.extractCSI(path)
csi = csip.reshape224x1(csi)
am = csip.extractAm(csi)
print(am.shape)