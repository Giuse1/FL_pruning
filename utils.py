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
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

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
        trainloader_list.append(torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader_list, testloader





