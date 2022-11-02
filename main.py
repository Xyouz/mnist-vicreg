import argparse

import torch
from torchvision import datasets, transforms
from model import encoder, projector, test_head
import torch.utils.data as data
from dataloader import VicRegDataset, TransformedDataset
import torch.optim as optim
from training import epoch_ssl, epoch_supervised, test_supervised

parser = argparse.ArgumentParser('VicReg for MNIST')

parser.add_argument(
    '-r',
    '--root',
    help='dataset root folder',
    type=str
)

parser.add_argument(
    '-d',
    '--device',
    help='torch device to use',
    type=str,
    default='cpu'
)

parser.add_argument(
    '-b',
    '--batch',
    type=int,
    default=64,
    help='batch size'
)

parser.add_argument(
    '-n',
    '--n_workers',
    type=int,
    default=1,
    help='number of workers to use for dataloading'
)

parser.add_argument(
    '-e',
    '--epochs',
    type=int,
    default=1,
    help='number of epochs'
)

args = parser.parse_args()

batch_size = args.batch
device = torch.device(args.device)
root = args.root
n_workers = args.n_workers
epochs = args.epochs

mnist = datasets.MNIST(root)
n_item = len(mnist)
mnist_ssl, mnist_test = data.random_split(mnist, [n_item * 9 // 10, n_item // 10])

vicreg_trnsfrm = transforms.Compose([
        transforms.RandomPerspective(0.25),
        transforms.RandomRotation(30, expand=True),
        transforms.RandomAutocontrast(0.25),
        transforms.RandomResizedCrop(28,(0.75,1.0)),
        transforms.ToTensor(),
        ])

mnist_vicreg = VicRegDataset(mnist_ssl, vicreg_trnsfrm)

vicregLoader = data.DataLoader(mnist_vicreg, batch_size, shuffle=True, num_workers=n_workers)

supervised_trnsfrm = transforms.Compose([
    transforms.RandomRotation(25),
    transforms.RandomResizedCrop(28, (0.75,1.0)),
    transforms.ToTensor()
])

supervisedData = TransformedDataset(mnist_test, supervised_trnsfrm)
supervisedLoader = data.DataLoader(supervisedData, batch_size, shuffle=True, num_workers=n_workers)

mnist_val = datasets.MNIST(root, train=False, transform=transforms.ToTensor())
val_loader = data.DataLoader(mnist_val, batch_size, num_workers=n_workers)

encoder = encoder.to(device)
projector = projector.to(device)
test_head = test_head.to(device)

opt_encoder = optim.SGD(encoder.parameters(), lr=0.01)
opt_projector = optim.SGD(projector.parameters(), lr=0.01)
opt_head = optim.SGD(test_head.parameters(), lr=0.01)

for i in range(epochs):
    print("SSL training (epoch {})".format(i))
    epoch_ssl(encoder, projector, opt_encoder, opt_projector, vicregLoader, device)
    if i % 5 == 0:
        print('Testing on classification')
        for _ in range(5):
            epoch_supervised(encoder, test_head, opt_head, supervisedLoader, device)
        test_supervised(encoder, test_head, val_loader, device)