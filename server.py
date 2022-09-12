import copy
from collections import OrderedDict
import torch.nn.utils.prune as prune
import torch
import logging
import utils


class Server(object):

    def __init__(self, client_type_dict, starting_dict):

        self.to_recover = [k for k, v in client_type_dict.items() if v == "bronze"]
        self.not_to_recover = [k for k, v in client_type_dict.items() if v == "gold"]
        self.starting_dict = starting_dict

        self.original_keys = list(starting_dict.keys())
        self.original_keys = [k.replace(".bias", '') for k in self.original_keys]
        self.original_keys = [k.replace(".weight", '') for k in self.original_keys]
        self.original_keys = list(dict.fromkeys(self.original_keys))
        self.present_rows_setted = False


    def set_present_rows(self):

        path = "/content/drive/MyDrive/"
        # path = ""

        self.present_rows_setted = True

        self.nonzero_output = {k: torch.load(f"{path}nonzero_indices/output/{k.replace('.weight', '')}.pt") for k in
                               self.starting_dict.keys() if "weight" in k}

        self.present_rows = [v.nonzero(as_tuple=True)[0].tolist() for v in self.nonzero_output.values()]
        self.present_rows.insert(0, [0, 1, 2])

        self.present_rows = self.present_rows[:11]

        self.input_nz, self.output_nz = {}, {}
        for idx_k, k in enumerate(self.original_keys):
            self.input_nz[k] = torch.load(f"{path}nonzero_indices/input/{k}.pt")
            self.output_nz[k] = torch.load(f"{path}nonzero_indices/output/{k}.pt")

    def model_evaluation(self, model, dataloader, criterion):
        with torch.no_grad():
            model.eval()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            running_loss = 0.0
            running_corrects = 0
            running_total = 0

            for (i, data) in enumerate(dataloader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                running_total += labels.shape[0]
                # break # todo

            epoch_loss = running_loss / running_total
            epoch_acc = running_corrects / running_total

            return epoch_loss, epoch_acc

    def average_weights(self, w, samples_per_client):

        for i in self.to_recover:
            w[i] = self.recover_matrix(w[i], 10, w[self.not_to_recover[0]])

        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(0, len(w)):
                if i == 0:
                    w_avg[key] = torch.true_divide(w[i][key], 1 / samples_per_client[i])
                else:
                    w_avg[key] += torch.true_divide(w[i][key], 1 / samples_per_client[i])
            w_avg[key] = torch.true_divide(w_avg[key], sum(samples_per_client.values()))
        return w_avg


    def recover_matrix(self, model_dict, n_classes, original_dict):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        reconstruced_dict = OrderedDict()

        for idx_k, k in enumerate(self.original_keys):

            dim = original_dict[f"{k}.weight"].shape
            reconstructed_w = torch.zeros(dim)

            if "features" in k:

                for idx_r, r in enumerate(self.present_rows[idx_k + 1]):
                    # reconstructed_b[r] = model_dict[k + ".bias"][idx_r]

                    for idx_c, c in enumerate(self.present_rows[idx_k]):
                        tmp = model_dict[k + ".weight"][idx_r, idx_c]

                        reconstructed_w[r, c] = tmp
                reconstruced_dict[k + ".weight"] = reconstructed_w.to(device)
                # reconstruced_dict[k + ".bias"] = reconstructed_b
            elif "classifier" in k and "classifier.6" not in k:

                reconstructed_b = torch.zeros(dim[0])

                mask = torch.zeros(dim)

                mask[self.output_nz[k], :] += torch.ones((int(dim[0] / 2), dim[1]))
                mask[:, self.input_nz[k]] += torch.ones((dim[0], int(dim[1] / 2)))
                mask = mask == 2

                reconstructed_b[self.output_nz[k]] = model_dict[k + ".bias"].cpu()
                reconstruced_dict[k + ".bias"] = reconstructed_b.to(device)

                reconstructed_w[mask] = torch.reshape(model_dict[k + ".weight"], (-1,)).cpu()
                reconstruced_dict[k + ".weight"] = reconstructed_w.to(device)

        k = "classifier.6"
        reconstruced_dict[k + ".bias"] = model_dict["classifier.6.bias"]
        dim = (n_classes, self.input_nz[k].shape[0])
        reconstructed_w = torch.zeros(dim)

        mask = torch.zeros(dim)
        mask[:, self.input_nz[k]] += torch.ones((dim[0], int(dim[1] / 2)))
        mask = mask == 1
        reconstructed_w[mask] = torch.reshape(model_dict[k + ".weight"], (-1,)).cpu()
        reconstruced_dict[k + ".weight"] = reconstructed_w.to(device)

        return reconstruced_dict
