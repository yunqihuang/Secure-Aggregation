import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from server import *
from client import *
from model import ResNet, MLP
import copy
from utils import *
import multiprocessing


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
        self.model = MLP(in_features=784, num_classes=10, hidden_dim=20)

    def init_clients(self):
        idx = self.sample()
        for i in range(self.clientNum):
            sub = DatasetSplit(self.trainSet, idx[i])
            # print(len(idx[i]), len(sub))
            c = WorkerProcess(i, sub, self.testSet, copy.deepcopy(self.model))
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
        self.init_clients()
        for j in range(self.clientNum):
            self.clients[j].start()


class WorkerProcess(multiprocessing.Process):
    def __init__(self, i, traindata, testdata, model):
        multiprocessing.Process.__init__(self)
        self.id = i
        self.local_dataSet = traindata
        self.testSet = testdata
        self.trainloader = DataLoader(self.local_dataSet, batch_size=64, shuffle=True)
        self.testloader = DataLoader(self.testSet, batch_size=64, shuffle=True)
        self.model = model
        self.epoch = 8
        self.local_epoch = 1
        self.lr = 0.01
        self.device = 'cpu'

    def local_update(self):
        self.model.train()
        epoch_loss = []
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)
        for i in range(self.local_epoch):
            batch_loss = []
            for img, target in self.trainloader:
                img = img.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(img)
                loss = loss_fn(outputs, target)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def evaluation(self, model):
        model.eval()
        loss_fn = nn.CrossEntropyLoss()
        loss, total, correct = 0.0, 0.0, 0.0
        for img, target in self.testloader:
            img = img.to(self.device)
            target = target.to(self.device)
            outputs = model(img)
            loss += loss_fn(outputs, target).item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, target)).item()
            total += len(target)

        accuracy = correct / total
        return accuracy, loss

    def run(self):
        avg_time = []
        for i in range(self.epoch):
            # print('worker {0} start epoch {1}\n'.format(self.id, i))
            w, loss = self.local_update()
            client = SecAggClient(self.id, 'http://127.0.0.1:5000')
            client.setDrop(0)
            client.create_handler()
            localModel, shape, nums = model2array(w)
            start_time = time.time()
            # print('worker {0} start secure aggregation\n local model:{1}\n'.format(self.id, localModel))
            res = client.start(localModel, i)
            # print('worker {0} end epoch {1}\n globalModel:{2}\n'.format(self.id, i, res))
            end_time = time.time()
            avg_time.append(end_time - start_time)
            globalModel = array2model(res['model'], shape, nums)
            self.model.load_state_dict(globalModel)
            # if self.id == 0 and i % 2 == 0:
            #     accuracy, loss = self.evaluation(self.model)
            #     print('epoch:{0} accuracy: {1}, communication timecost:{2}'
            #           .format(i, accuracy, end_time - start_time))
        if self.id == 0:
            print(np.sum(avg_time)/len(avg_time))
        return


if __name__ == '__main__':
    x = FedAvg(30)
    x.start()
