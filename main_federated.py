# todo non dividere per 4 tutta la matrice, ma anche per due dove i due client hanno gli zeri
# todo break
# todo bias
import torch
import torch.nn as nn
import torchvision
import utils
from FL_train import train_model
import vgg11_custom
import os

import torch.nn as nn
import torch.nn.functional as F




if os.path.exists("/content/drive/MyDrive/"):
    path = "/content/drive/MyDrive/"
else:
    path = ""

pruning_percentage = 0.25

utils.set_seed(0)

total_num_users = 4
num_users = 4
local_epochs = 1
lr = 0.01

num_rounds = 100
batch_size = 64  # todo
in_size = 224  # todo

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = torchvision.models.vgg11(pretrained=False)
model = vgg11_custom.vgg11(pretrained=False, rate=1)
in_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(in_features, out_features=10, bias=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

client_type_dict = {0: "gold", 1:"gold", 2:"bronze", 3:"bronze"}

name_log = "bronze"
train_loss, train_acc, val_loss, val_acc = train_model(name_log, model, criterion, num_rounds=num_rounds,
                                                       local_epochs=local_epochs, total_num_users=total_num_users,
                                                       num_users=num_users, batch_size=batch_size, learning_rate=lr,
                                                       iid=False, in_size=in_size, client_type_dict=client_type_dict,
                                                       path=path,pruning_percentage=pruning_percentage)
