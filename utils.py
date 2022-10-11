import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from operator import itemgetter
import random

def set_seed(seed):
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def get_cifar_iid(batch_size, total_num_clients, in_size):
    mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
    transform = transforms.Compose([
        transforms.Resize(in_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    total_data = len(trainset)
    random_list = random.sample(range(total_data), total_data)
    data_per_client = int(total_data / total_num_clients)
    datasets = []
    for i in range(total_num_clients):

        indexes = random_list[i*data_per_client: (i+1)*data_per_client]
        # datasets.append(list(itemgetter(*indexes)(trainset)))
        datasets.append(torch.utils.data.Subset(trainset, indexes))
    trainloader_list = []
    for d in datasets:
        trainloader_list.append(torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader_list, testloader

def get_cifar_noniid(batch_size, total_num_clients, in_size):
    mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
    transform = transforms.Compose([
        transforms.Resize(in_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    indices = [idx for idx, target in enumerate(trainset.targets) if target in range(7)]
    trainset_gold = torch.utils.data.Subset(trainset, indices)

    indices = [idx for idx, target in enumerate(trainset.targets) if target in [7,8,9]]
    trainset_bronze = torch.utils.data.Subset(trainset, indices)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    trainset_gold1, trainset_gold2 = torch.utils.data.random_split(trainset_gold, [int(35000/2), int(35000/2)],
                                                                          generator=torch.Generator().manual_seed(42))
    trainset_bronze1, trainset_bronze2 = torch.utils.data.random_split(trainset_bronze, [int(15000/2), int(15000/2)],
                                                                   generator=torch.Generator().manual_seed(42))

    train_list = [ trainset_gold1, trainset_gold2,trainset_bronze1, trainset_bronze2 ]
    trainloader_list = [torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True) for d in train_list]

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader_list, testloader


def get_cifar100_subset(batch_size, total_num_clients, in_size):
    mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
    transform = transforms.Compose([
        transforms.Resize(in_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    trainset_cifar_100 = torchvision.datasets.CIFAR100("~/data", download=True, train=True, transform=transform)
    testset_cifar_100 = torchvision.datasets.CIFAR100("~/data", download=False, train=False, transform=transform)

    all_classes = trainset_cifar_100.classes

    classes_to_keep = ["bicycle", "bus", "motorcycle", "train", "tractor"]
    indeces_class_to_keep = [all_classes.index(e) for e in classes_to_keep]

    indices = [idx for idx, target in enumerate(trainset_cifar_100.targets) if target in indeces_class_to_keep]
    trainset = torch.utils.data.Subset(trainset_cifar_100, indices)

    indices = [idx for idx, target in enumerate(testset_cifar_100.targets) if target in indeces_class_to_keep]
    testset = torch.utils.data.Subset(testset_cifar_100, indices)

    dict_new_class = {}
    for i, x in enumerate(indeces_class_to_keep):
        dict_new_class[x] = i + 9 + 1

    trainset =  [(x[0], dict_new_class[x[1]]) for x in trainset]
    testset = [(x[0], dict_new_class[x[1]]) for x in testset]




    total_data = len(trainset)
    random_list = random.sample(range(total_data), total_data)
    data_per_client = int(total_data / total_num_clients)
    datasets = []
    for i in range(total_num_clients):
        indexes = random_list[i * data_per_client: (i + 1) * data_per_client]
        # datasets.append(list(itemgetter(*indexes)(trainset)))
        datasets.append(torch.utils.data.Subset(trainset, indexes))
    trainloader_list = []
    for d in datasets:
        trainloader_list.append(
            torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4,
                                             pin_memory=True)

    return trainloader_list, testloader






