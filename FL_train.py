import copy
import random
import utils
from client import ClientGold, ClientBronze
from server import Server
from utils import get_cifar_iid

random.seed(0)


def train_model(global_model, criterion, num_rounds, local_epochs, total_num_users, num_users, batch_size,
                learning_rate, iid, in_size, client_type_dict):
    utils.set_seed(0)

    print(client_type_dict)

    server = Server(client_type_dict, copy.deepcopy(global_model.state_dict()))




    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    if iid:
        trainloader_list, valloader = get_cifar_iid(batch_size=batch_size, total_num_clients=total_num_users,
                                                    in_size=in_size)

    clients = {}
    for k, v in client_type_dict.items():
        if v == "gold":
            clients[k] = ClientGold(dataloader=trainloader_list[k], id=k, criterion=criterion,
                                local_epochs=local_epochs, learning_rate=learning_rate)
        else:
            clients[k] = ClientBronze(dataloader=trainloader_list[k], id=k, criterion=criterion,
                                    local_epochs=local_epochs, learning_rate=learning_rate, in_size=in_size)
    for round in range(num_rounds):
        print('-' * 10)
        print('Epoch {}/{}'.format(round, num_rounds - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                local_weights = {}
                samples_per_client = {}

                random_list = random.sample(range(total_num_users), num_users)

                for idx in random_list:

                    local_state_dict, local_n_samples = clients[idx].update_weights(model=copy.deepcopy(global_model), epoch=round)

                    local_weights[idx] = copy.deepcopy(local_state_dict)
                    samples_per_client[idx] = (local_n_samples)
                    print(clients[idx])
                    print(local_weights[idx]["features.0.weight"].shape)

                # random list need to order th models accoridng to bronze and gold
                if not server.present_rows_setted:
                    server.set_present_rows()

                global_weights = server.average_weights(local_weights, samples_per_client)
                global_model.load_state_dict(global_weights)

            else:
                val_loss_r, val_accuracy_r = server.model_evaluation(model=global_model,
                                                                     dataloader=valloader, criterion=criterion)
                val_loss.append(val_loss_r)
                val_acc.append(val_accuracy_r)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, val_loss_r, val_accuracy_r))

    return train_loss, train_acc, val_loss, val_acc
