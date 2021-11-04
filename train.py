import _thread

from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from server import *
from client import *
from model import ResNet


def getdata():
    apply_transform = transforms.Compose(
        [transforms.Resize([224, 224]),
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
        self.iid = True

    def init_clients(self):
        idx = self.sample()
        for i in range(self.clientNum):
            sub = Subset(self.trainSet, idx[i])
            # print(len(idx[i]), len(sub))
            self.local_trainSet.append(sub)

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

