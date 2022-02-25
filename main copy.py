import csiread
import numpy as np
import csitools
import cv2

# Чтение csi
path = './csi_data/homelocation/r_oomk_itchent_oilet/train/room.dat'
data = csiread.Atheros(path, nrxnum=2, ntxnum=5, tones=56, if_report=False)
data.read(endian='big')
csi = data.csi[(data.payload_len == 556) & (data.nc == 2)][:, :, :2, :2]
ampl = np.abs(csi)

ampl = np.reshape(ampl, (ampl.shape[0], ampl.shape[1], -1))
ampl = np.transpose(ampl, (0, 2, 1))
ampl = np.reshape(ampl, (ampl.shape[0], -1))
print(ampl[0])

# cv2.imshow('wew', ampl[:1000,:] / ampl.max())
# Create a Named Window
win_name = 'one win'
# cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# # Move it to (X,Y)
# # cv2.moveWindow(win_name, X, Y)
    
# # Show the Image in the Window
# cv2.imshow(win_name, ampl[:10000,:] / ampl.max())
    
# # Resize the Window
# cv2.resizeWindow(win_name, 1920, 1080)
    
# # Wait for <> miliseconds
# cv2.waitKey(0)
print('shape:', ampl.shape)
ampl = ampl[:30000,:] / ampl[:30000,:].max() * 255
cv2.imwrite('./test.jpg', ampl)

array = np.arange(0, 737280, 1, np.uint8)
array = np.reshape(array, (1024, 720))

print(ampl.shape, ampl.max())
print(array.shape, array.max())
cv2.imwrite('./filename.jpeg', array)