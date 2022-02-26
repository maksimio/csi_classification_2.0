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
dirpath = './csi_data/use_in_paper/2_objects/train'
frameHeight = 50
y = None
x = None

for fname in listdir(dirpath):
  csi = csir.extractCSI(dirpath + '/' + fname)
  csi = csip.extractAm(csi)
  csi = csip.reshape4x56(csi)
  # csi = np.diff(csi)

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
batchSize = 30

net = csim.FirstNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
  for i in range(0, len(x) - batchSize, batchSize):
    inputs = torch.from_numpy(x[i:i+batchSize]).double()
    labels = torch.from_numpy(y[i:i+batchSize]).double()
    print(inputs)
    optimizer.zero_grad()

        # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 2000 == 1999:    # print every 2000 mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0


print('Finished Training')