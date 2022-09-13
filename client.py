import logging
from collections import OrderedDict
import torch
import torch.nn.utils.prune as prune
# from simplify import simplify
from simplify.simplify import simplify



class ClientGold(object):

    def __init__(self, dataloader, id, criterion, local_epochs, learning_rate):

        # path = "/content/drive/MyDrive/"
        path = ""
        self.id = id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.local_epochs = local_epochs
        self.dataloader = dataloader
        self.learning_rate = learning_rate

        self.logger = logging.getLogger(f'client{id}')
        self.logger.setLevel(logging.INFO)

        ch = logging.FileHandler(f'{path}reports/client{id}', "w")
        ch.setLevel(logging.INFO)
        self.logger.addHandler(ch)
        self.logger.info("local_loss,local_correct,len_dataset")

    def update_weights(self, model, epoch):
        model.train()
        lr = self.learning_rate
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for _ in range(self.local_epochs):

            local_correct = 0
            local_loss = 0.0
            for (i, data) in enumerate(self.dataloader):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                _, preds = torch.max(log_probs, 1)
                local_correct += torch.sum(preds == labels).cpu().numpy()

                loss.backward()
                optimizer.step()
                local_loss += loss.item() * images.size(0)
                # break # todo

        self.logger.info(f"{local_loss},{local_correct},{len(self.dataloader.dataset)}")

        return model.state_dict(), len(self.dataloader.dataset)


class ClientBronze(object):

    def __init__(self, dataloader, id, criterion, local_epochs, learning_rate, in_size, pruning_percentage=0.5):
        self.id = id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.local_epochs = local_epochs
        self.dataloader = dataloader
        self.learning_rate = learning_rate

        self.logger = logging.getLogger(f'client{id}')
        self.logger.setLevel(logging.INFO)

        # path = "/content/drive/MyDrive/"
        path = ""
        ch = logging.FileHandler(f'{path}reports/client{id}', "w")
        ch.setLevel(logging.INFO)
        self.logger.addHandler(ch)
        self.logger.info("local_loss,local_correct,len_dataset")

        self.mask_created = False
        self.in_size = in_size
        self.pruning_percentage = pruning_percentage

    def update_weights(self, model, epoch):

        if not self.mask_created:
            self.mask_dict = self.prune_model_fixed(model=model, percentage=self.pruning_percentage, device=self.device,
                                                    in_size=self.in_size)
            self.mask_created = True
        else:
            self.prune_from_mask(model, self.device, self.in_size)

        model.train()
        lr = self.learning_rate
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for _ in range(self.local_epochs):

            local_correct = 0
            local_loss = 0.0
            for (i, data) in enumerate(self.dataloader):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                _, preds = torch.max(log_probs, 1)
                local_correct += torch.sum(preds == labels).cpu().numpy()

                loss.backward()
                optimizer.step()
                local_loss += loss.item() * images.size(0)
                # break # todo

        self.logger.info(f"{local_loss},{local_correct},{len(self.dataloader.dataset)}")

        return model.state_dict(), len(self.dataloader.dataset)

    def prune_model_fixed(self, model, percentage, device, in_size):

        parameters_to_prune = (
            ("features.0", 'weight'),
            ("features.3", 'weight'),
            ("features.6", 'weight'),
            ("features.8", 'weight'),
            ("features.11", 'weight'),
            ("features.13", 'weight'),
            ("features.16", 'weight'),
            ("features.18", 'weight'),
            ("classifier.0", 'weight'),
            ("classifier.3", 'weight'))

        mask_dict = OrderedDict()
        for name, attr in parameters_to_prune:
            layer_name, n = name.split(".")
            prune.ln_structured(getattr(model, layer_name)[int(n)], name=attr,
                                amount=percentage, n=2, dim=0)  # pp_prune[f"{layer_name}.{n}.{attr}"]
            mask_dict[name] = model.state_dict()[name + ".weight_mask"]
            prune.remove(getattr(model, layer_name)[int(n)], name=attr)

        dummy_input = torch.zeros(1, 3, in_size, in_size).to(device)
        simplify(model, dummy_input)

        return mask_dict

    def prune_from_mask(self, model, device, in_size):

        for name, mask in self.mask_dict.items():
            cmp, idx = name.split('.')
            prune.custom_from_mask(getattr(model, cmp)[int(idx)], name="weight", mask=mask)
            prune.remove(getattr(model, cmp)[int(idx)], 'weight')
            # torch.nan_to_num(model.state_dict()[f"{cmp}.{int(idx)}.weight"], nan=0.5)

        dummy_input = torch.zeros(1, 3, in_size, in_size).to(device)
        simplify(model, dummy_input)





