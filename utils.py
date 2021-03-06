import torch
from collections import OrderedDict
from torch.utils.data import Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


def model2array(m):
    shapes = {}
    nums = []
    res = torch.tensor([])
    for key in m.keys():
        x = m[key]
        count = torch.numel(x)
        shapes[key] = x.size()
        nums.append(count)
        p = x.view(1, count)
        res = torch.cat((res, p), 1)
    return res.numpy()[0], shapes, nums


def array2model(array, shapes, nums):
    res = OrderedDict()
    p = 0
    i = 0
    for k, v in shapes.items():
        t = torch.tensor(array[p:(p + nums[i])])
        t = t.view(v)
        res[k] = t
        p = p + nums[i]
        i = i + 1
    return res
