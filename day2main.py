"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# class my_dataset(Dataset):
#     def __init__(self, path, preprocess):
#         self.preprocess = preprocess
#         self.image_paths = []
#         self.labels = []
#         label_list = os.listdir(path)
#         for label in label_list:
#             image_folder = os.path.join(path, label)
#             for file_names in os.listdir(image_folder):
#                 if file_names.endswith(("png", "jpg", "jpeg")):
#                     self.image_paths.append(os.path.join(image_folder, file_names))
#                     self.labels.append(label_list.index(label))
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, item):
#         image = Image.open(self.image_paths[item])
#         image = self.preprocess(image)
#         label = self.labels[item]
#         return image, label


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
# print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# class own_dataset(Dataset):
#     # init,len,getitem
#     def __init__(self, root, preprocess):
#         super(own_dataset, self).__init__()
#         self.preprocess = preprocess
#         self.image_paths = []
#         self.labels = []
#         label_list = os.listdir(root)
#         for label in label_list:
#             # root=/home label=apple os.path.join(root,label):/home/apple
#             image_folder = os.path.join(root, label)
#             for file in os.listdir(image_folder):
#                 if file.endswith(("png", "jpg", "gif")):
#                     self.image_paths.append(os.path.join(image_folder, file))
#                     self.labels.append(label_list.index(label))
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, item):
#         image = Image.open(self.image_paths[item])
#         image = self.preprocess(image)
#         label = self.labels[item]
#         return image, label
#
#     def print_len(self):
#         print(len(self.image_paths))
#
#
# trainset = own_dataset(root="./data", preprocess=transform_train)


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
# print("训练集样本数:", len(trainset))  # 输出: 50000
# print("测试集样本数:", len(testset))   # 输出: 10000
# print("类别标签:", trainset.classes)  # 输出: ['airplane', 'automobile', ...]
# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
def create_model():
    # 加载预训练权重
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

    # 冻结除最后一层外的所有参数
    for param in net.parameters():
        param.requires_grad = False

    # 替换最后一层（ImageNet1000类→CIFAR10的10类）
    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, 10)
    net.fc.requires_grad = True  # 确保最后一层梯度开启

    return net
net = create_model().to(device)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    # weight = net.state_dict()
    # torch.save(weight, "/your/path")
    # weight = torch.load("/your/path")
    # net.load_state_dict(weight)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)  # loss=L+\lambda||w||^2
optimizer = optim.SGD(
    net.fc.parameters(),  # 仅优化最后一层
    lr=args.lr,
    momentum=0.9,
    weight_decay=5e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()#启动训练模式！
    train_loss = 0
    correct = 0
    total = 0
    train_bar = tqdm(trainloader,desc="Training", leave=True)
    for batch_idx, (inputs, targets) in enumerate(train_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad() # 清除所有参数的梯度
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # for param in net.parameters():
        #     print(param.data,param.grad)
        optimizer.step()# 更新所有参数（权重和偏置）
        _, predicted = outputs.max(1)
        total += targets.size(0)#已经处理的图片数
        correct += predicted.eq(targets).sum().item()

        # tqdm
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        train_bar.set_postfix(
            Loss=f"{train_loss / (batch_idx + 1):.3f}",
            Acc=f"{100. * correct / total:.3f}%")

    return train_loss / len(trainloader),100. * correct / total


def test(epoch):
    global best_acc
    net.eval()
    # for param in net.parameters():
    #     param.requires_grad = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        test_bar = tqdm(testloader, desc="Testing", leave=True)
        for batch_idx, (inputs, targets) in enumerate(test_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            test_bar.set_postfix(
                Loss=f"{test_loss / (batch_idx + 1):.3f}",
                Acc=f"{100. * correct / total:.3f}%")
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return test_loss / len(testloader),acc

train_epoch_loss = []
test_epoch_loss = []
train_epoch_acc = []
test_epoch_acc = []
for epoch in range(start_epoch, start_epoch + 200):
    train_loss,train_acc = train(epoch)
    train_epoch_loss.append(train_loss)
    train_epoch_acc.append(train_acc)
    test_loss,test_acc = test(epoch)
    test_epoch_loss.append(test_loss)
    test_epoch_acc.append(test_acc)
    scheduler.step()

plt.figure()
plt.subplot(121)
plt.plot(train_epoch_loss,label = 'train_loss')
plt.plot(test_epoch_loss,label ='test_loss')
plt.xlabel('epoch')
plt.ylabel("loss")
plt.legend()
plt.subplot(122)
plt.plot(train_epoch_loss,label = "train_acc")
plt.plot(test_epoch_acc,label = 'test_acc')
plt.xlabel('epoch')
plt.ylabel("acc")
plt.legend()
plt.show()


# 使用tqdm改写代码中的进度条功能
# 添加绘制图像代码，绘制损失和正确率随着epoch的折线图(matplotlib)
# https://www.robots.ox.ac.uk/~vgg/data/pets/下载Pets数据集，编写相应的数据集加载代码
# 使用torchvision的resnet18模型及其权重，冻结除了最后一层之外的所有参数，只训练最后一层。（在大规模数据集上学习到的特征提取器可以迁移到小数据集上）
