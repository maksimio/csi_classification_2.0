from os import listdir
from re import X
from csitools import reading
import csitools.processing as csip
import csitools.reading as csir
import csitools.viewing as csiv
import csitools.models as csim
import numpy as np
import torch

categories = ['air', 'bottle']
dirpath = './csi_data/use_in_paper/2_objects/test'
frameHeight = 50
y = None
x = None

for fname in listdir(dirpath):
  csi = csir.extractCSI(dirpath + '/' + fname)
  csi = csip.extractAm(csi)
  csi = csip.reshape4x56(csi)
  csi = np.diff(csi)

  csi = np.expand_dims(csi, axis=0)
  # print(csi.shape)
  csi = csi[:,:csi.shape[1]//frameHeight*frameHeight,:,:]
  # print(csi.shape)
  csi = np.reshape(csi, (-1, frameHeight, 4, csi.shape[3]))
  # print(csi.shape)
  csi = np.transpose(csi, (0,2,1,3))
  # print(csi.shape)
  rep = np.repeat(csir.match(categories, fname), csi.shape[0])
  if type(y) != np.ndarray:
    y = rep
    x = csi
  else:
    y = np.concatenate((y, rep))
    x = np.concatenate((x, csi))

def shuffle_in_unison(a,b):
    assert len(a)==len(b)
    c = np.arange(len(a))
    np.random.shuffle(c)
    return a[c],b[c]

x,y = shuffle_in_unison(x,y)
print(y.shape, x.shape)
print(x[0].shape)
batchSize = 20

net = csim.FirstNet()
PATH = './my_first.pth'
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for i in range(0, len(x) - batchSize, batchSize):
        inputs = torch.from_numpy(x[i:i+batchSize]).float()
        labels = torch.from_numpy(y[i:i+batchSize])
        # calculate outputs by running images through the network
        outputs = net(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test inputs: {100 * correct // total} %')

