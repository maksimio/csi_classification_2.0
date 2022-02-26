import csitools.processing as csip
import csitools.reading as csir
import csiread

path = './csi_data/use_in_paper/2_objects/train/D=2020-01-02_T=21-09-35--air.dat'
csi = csir.extractCSI(path)
am = csip.extractAm(csi)

