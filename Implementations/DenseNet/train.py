import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tfms

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from models_cifar10 import densenet_bc_100


"""Parameters"""
save_dir = './model_save'
os.makedirs(save_dir, exist_ok=True)

epochs = 300
batch_size = 64
initial_lr = 0.1
valid_ratio = 0.05
print_frequency = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

train_set = CIFAR10(root='./data/',
                  download=True,
                  transform=tfms.Compose([
                      tfms.ToTensor(),
                      tfms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                  ]),
                  train=True)

dataset_size = len(train_set)
valid_size = int(valid_ratio * dataset_size)
train_set, valid_set = random_split(train_set, [dataset_size-valid_size, valid_size])

test_set = CIFAR10(root='./data/',
                  download=True,
                  transform=tfms.Compose([
                      tfms.ToTensor(),
                      tfms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                  ]),
                  train=False)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)


model = densenet_bc_100().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
schedular = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                           milestones=[int(epochs * 0.5), int(epochs * 0.75)],
                                           gamma=0.1, last_epoch=-1)

best_acc= 0.0
for epoch in range(1, epochs+1):
    avg_loss = 0
    model.train()
    for i, datas in enumerate(train_loader):
        inps, labels = datas
        inps, labels = inps.to(device), labels.to(device)

        outs = model(inps)
        loss = criterion(outs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        if i % print_frequency == (print_frequency-1) or i == 0:
            template = "[{:3d}/{:3d}]  [{:5d}/{:5d}]  [Loss: {:.6f}]".format(
                epoch, epochs,
                (i+1), len(train_loader),
                avg_loss / (i+1)
            )
            print(template)


    """ Validation """
    correct = 0
    total   = 0

    model.eval()
    for i, datas in enumerate(valid_loader):
        inps, labels = datas
        inps, labels = inps.to(device), labels.to(device)

        outs = model(inps)
        _, preds = torch.max(outs.data, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        template = "[{:3d}/{:3d}]  [{:5d}/{:5d}]  [Correct: {:.2f}% | {:4d}/{:4d}]".format(
            epoch, epochs,
            (i + 1), len(valid_loader),
            correct/total*100,
            correct, total
        )
        print(template)

    if (correct / total > best_acc):
        torch.save(os.path.join(save_dir,
                                "densenet_40_{:.2f}.ptr".format(
                                    best_acc*100)),
                   model.state_dict())

    schedular.step()