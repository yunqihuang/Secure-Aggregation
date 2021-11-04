import _thread

import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from server import *
from client import *
from model import ResNet, MLP
import copy
from utils import model2array, array2model


def getdata():
    apply_transform = transforms.Compose(
        [
            # transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=apply_transform)
    test_data = datasets.MNIST('./data', train=False, download=False, transform=apply_transform)
    return train_data, test_data


class FedAvg:
    def __init__(self, n):
        self.clientNum = n
        self.trainSet, self.testSet = getdata()
        self.local_trainSet = []
        self.clients = []
        self.iid = True
        self.model = MLP()

    def init_clients(self):
        idx = self.sample()
        for i in range(self.clientNum):
            sub = Subset(self.trainSet, idx[i])
            # print(len(idx[i]), len(sub))
            c = Worker(i, sub, copy.deepcopy(self.model))
            self.clients.append(c)

    def sample(self):
        if self.iid:
            num_items = int(len(self.trainSet) / self.clientNum)
            dict_users, all_idxs = {}, [i for i in range(len(self.trainSet))]
            for i in range(self.clientNum):
                dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                                     replace=False))
                all_idxs = list(set(all_idxs) - dict_users[i])
            return dict_users
        else:
            return []

    def start(self):
        epoch = 5
        for i in range(epoch):
            for j in range(self.clientNum):
                pass


class Worker:
    def __init__(self, i, dataset, model):
        self.id = i
        self.local_dataSet = dataset
        self.dataloader = DataLoader(self.local_dataSet, batch_size=64, shuffle=True)
        self.model = model
        self.local_epoch = 5
        self.lr = 0.01
        self.device = 'cpu'

    def local_update(self):
        self.model.train()
        epoch_loss = []
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)
        for i in range(self.local_epoch):
            batch_loss = []
            for img, target in self.dataloader:
                img = img.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                logits = self.model(img)
                loss = loss_fn(logits, target)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def step(self, epoch):
        w, loss = self.local_update()
        client = SecAggClient(self.id)
        client.setDrop(0)
        client.create_handler()
        localModel, shape, nums = model2array(w)
        res = client.start(localModel, epoch)
        globalModel = array2model(res, shape, nums)
        self.model.load_state_dict(res)
        return


x, y, z = model2array(MLP(2, 2, 1).state_dict())
print(x.dtype)
# x = np.ones(10)
# print(x.dtype)
data = base64.b64encode(x)
r = base64.b64decode(data)
print(r)
model = np.frombuffer(r, dtype='float32')
print(model)