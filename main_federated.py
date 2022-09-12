# todo non dividere per 4 tutta la matrice, ma anche per due dove i due client hanno gli zeri
# todo break
# todo bias
from datetime import datetime

import torch
import torch.nn as nn
import torchvision

import utils
from FL_train import train_model
import vgg11_custom

utils.set_seed(0)

total_num_users = 2
num_users = 2
local_epochs = 1
lr = 0.01

num_rounds = 4  # todo
batch_size = 4  # todo
in_size = 32  # todo
#
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler(f"reports/fl.csv", mode='w')
#     ],
# )

#
# logging.info(f"total_num_users: {total_num_users}")
# logging.info(f"num_users: {num_users}")
# logging.info(f"num_rounds: {num_rounds}")
# logging.info(f"local_epochs: {local_epochs}")
# logging.info(f"batch_size: {batch_size}")
# logging.info(f"learning_rate: {lr}")

# logging.info("time,timestamp,cputime,operation")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = torchvision.models.vgg11(pretrained=False)
model = vgg11_custom.vgg11(pretrained=False)
in_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(in_features, out_features=10, bias=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

client_type_dict = {0: "gold", 1:"bronze"}

train_loss, train_acc, val_loss, val_acc = train_model(model, criterion, num_rounds=num_rounds,
                                                       local_epochs=local_epochs, total_num_users=total_num_users,
                                                       num_users=num_users, batch_size=batch_size, learning_rate=lr,
                                                       iid=True, in_size=in_size, client_type_dict=client_type_dict)
