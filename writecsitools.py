import csitools.processing as csip
import csitools.reading as csir
import csitools.viewing as csiv
import csitools.models as csim
import csitools
import numpy as np
import torch
from torch import utils

path = './csi_data/use_in_paper/2_objects/train/D=2020-01-02_T=21-09-35--air.dat'
csi = csir.extractCSI(path)
csi = csip.extractAm(csi)
csi = csip.reshape4x56(csi)
csi = np.diff(csi)
csi = np.reshape(csi, (csi.shape[0], -1))
csi = csip.down(csi)
# csiv.imsave('./newtest4.jpg', csi)

net = csim.FirstNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

trainLoader = utils.data.DataLoader(csi, batch_size=64, shuffle=True)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        print(data.shape)
        print(csi.shape)
        exit()
        # zero the parameter gradients
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