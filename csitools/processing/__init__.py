import numpy as np
import csiread

# Преобразование к виду 56 x 4 - амплитуды или фазы
def reshape56x4(csi: np.ndarray):
  # csi = csi.reshape()
  pass

# Преобразование к виду 224 x 1 - амплитуды или фазы
def reshape224x1(arr):
  pass

# Нарезка по первой размерности по заданному чанку (по умолчанию - квадрат).
# Неполный чанк в конце отбрасывается
def chunks(arr):
  pass

def extractAm(csi: np.ndarray):
  return np.abs(csi)

def extractPh(csi: np.ndarray):
  ph = np.unwrap(np.angle(csi), axis=1)
  return csiread.utils.calib(ph, k=csiread.utils.scidx(20, 1), axis=1)