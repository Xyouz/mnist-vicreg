from torch.utils.data import Dataset

class VicRegDataset(Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self.transform = transform
        self.MNISTDataset = dataset

    def __len__(self):
        return len(self.MNISTDataset)

    def __getitem__(self, item):
        item, number = self.MNISTDataset.__getitem__(item)
        transformed1 = self.transform(item)
        transformed2 = self.transform(item)
        return transformed1, transformed2

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data, target = self.dataset.__getitem__(item)
        data = self.transform(data)
        return data, target