import torchvision
import torch
import numpy as np
from advertorch.attacks import PGDAttack,CarliniWagnerL2Attack
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIR_PATH = os.path.dirname(os.path.abspath(__file__))

class MNISTdata():
    def __init__(self):
        self.num_channel = 1
        self.img_size = 28
        self.num_lables = 10
        data_train = torchvision.datasets.MNIST(DIR_PATH, train=True, download=True,
                                                transform=torchvision.transforms.ToTensor())
        data_test = torchvision.datasets.MNIST(DIR_PATH, train=False, download=True,
                                               transform=torchvision.transforms.ToTensor())

        self.train_data = torch.mul(data_train.data, 1 / 255).reshape((-1, 1, 28, 28)).type(torch.float32)
        self.train_labels = data_train.targets
        self.test_data = torch.mul(data_test.data, 1 / 255).reshape((-1, 1, 28, 28)).type(torch.float32)
        self.test_labels = data_test.targets


class dataSet(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.labels[index]
        return data, labels


class normalMnist():
    def __init__(self, data_type="test", loader_batch=128,shuffle=None):
        data = MNISTdata()
        if data_type == 'train':
            self.data = data.train_data
            self.labels = data.train_labels
            self.shuffle = True
        else:
            self.data = data.test_data
            self.labels = data.test_labels
            self.shuffle = False
        if shuffle is not None:
            self.shuffle = shuffle

        self.loader = torch.utils.data.DataLoader(dataSet(self.data, self.labels), batch_size=loader_batch,shuffle=self.shuffle)


class attackMnist():
    def __init__(self, attack_model, attack_method="FGSM", eps=0.3, data_type="test", rand_seed=0, rand_min=0,
                 rand_max=1, loader_batch=128, for_trainning=False, atk_loss=None, quantize=False,shuffle=None,attack_iter=10,attack_params={}):

        normal_data = normalMnist(data_type=data_type, loader_batch=loader_batch)
        self.noarmal_data = normal_data.data
        self.labels = normal_data.labels

        x_atk = torch.tensor([]).to(device)
        bs = loader_batch
        for batch_data, batch_labels in normal_data.loader:
            if (attack_method == "FGSM"):
                if isinstance(eps, str):
                    batch_pn = FGSM(attack_model, loss_fn=atk_loss, getAtkpn=True,**attack_params).perturb(batch_data.to(device),
                                                                                           batch_labels.to(device))
                    eps_temp = (1) * torch.rand((len(batch_pn), 1, 1, 1))
                    eps_temp = eps_temp.to(device)
                    batch_atk = torch.clamp(batch_data.to(device) + eps_temp * batch_pn, min=0, max=1)
                else:
                    batch_atk = FGSM(attack_model, loss_fn=atk_loss, eps=eps,**attack_params).perturb(batch_data.to(device),
                                                                                      batch_labels.to(device))
            elif (attack_method == "PGD"):
                batch_atk = PGDAttack(attack_model, loss_fn=atk_loss, eps=eps,**attack_params).perturb(batch_data.to(device),
                                                                                       batch_labels.to(device))
            elif attack_method == "CW":
                batch_atk = CarliniWagnerL2Attack(attack_model,num_classes=10,loss_fn=atk_loss, **attack_params).perturb(batch_data.to(device),batch_labels.to(device))
            elif attack_method == "iFGSM":
                batch_atk = batch_data
                for _ in range(attack_iter):
                    batch_atk = FGSM(attack_model, loss_fn=atk_loss, eps=eps/attack_iter, **attack_params).perturb(batch_atk.to(device), batch_labels.to(device))
            x_atk = torch.cat((x_atk, batch_atk))
        # x_atk = torch.tensor(x_atk)
        self.data = x_atk.cpu()
        if quantize:
            self.data = (self.data * 255).type(torch.int) / 255.
        
        self.shuffle = True if data_type=='train' else False
        if shuffle is not None:
            self.shuffle = shuffle

        if for_trainning:
            self.loader = torch.utils.data.DataLoader(train_dataSet(self.noarmal_data, self.labels, self.data),
                                                      batch_size=loader_batch,shuffle=self.shuffle)
        else:
            self.loader = torch.utils.data.DataLoader(dataSet(self.data, self.labels), batch_size=loader_batch,shuffle=self.shuffle)


class FMNISTdata():
    def __init__(self):
        self.num_channel = 1
        self.img_size = 28
        self.num_lables = 10
        data_train = torchvision.datasets.FashionMNIST(DIR_PATH, train=True, download=True,
                                                transform=torchvision.transforms.ToTensor())
        data_test = torchvision.datasets.FashionMNIST(DIR_PATH, train=False, download=True,
                                               transform=torchvision.transforms.ToTensor())

        self.train_data = torch.mul(data_train.data, 1 / 255).reshape((-1, 1, 28, 28)).type(torch.float32)
        self.train_labels = data_train.targets
        self.test_data = torch.mul(data_test.data, 1 / 255).reshape((-1, 1, 28, 28)).type(torch.float32)
        self.test_labels = data_test.targets




class normalFMnist():
    def __init__(self, data_type="test", loader_batch=128,shuffle=None):
        data = FMNISTdata()
        if data_type == 'train':
            self.data = data.train_data
            self.labels = data.train_labels
            self.shuffle = True
        else:
            self.data = data.test_data
            self.labels = data.test_labels
            self.shuffle = False
        
        if shuffle is not None:
            self.shuffle = shuffle

        self.loader = torch.utils.data.DataLoader(dataSet(self.data, self.labels), batch_size=loader_batch, shuffle=self.shuffle)


class attackFMnist():
    def __init__(self, attack_model, attack_method="FGSM", eps=0.3, data_type="test", rand_seed=0, rand_min=0,
                 rand_max=1, loader_batch=128, for_trainning=False, atk_loss=None, quantize=False,shuffle=None,attack_params={},attack_iter=10):

        normal_data = normalFMnist(data_type=data_type, loader_batch=loader_batch)
        self.noarmal_data = normal_data.data
        self.labels = normal_data.labels

        x_atk = torch.tensor([]).to(device)
        bs = loader_batch
        for batch_data, batch_labels in normal_data.loader:
            if (attack_method == "FGSM"):
                if isinstance(eps, str):
                    batch_pn = FGSM(attack_model, loss_fn=atk_loss, getAtkpn=True,**attack_params).perturb(batch_data.to(device),
                                                                                           batch_labels.to(device))
                    eps_temp = (1) * torch.rand((len(batch_pn), 1, 1, 1))
                    eps_temp = eps_temp.to(device)
                    batch_atk = torch.clamp(batch_data.to(device) + eps_temp * batch_pn, min=0, max=1)
                else:
                    batch_atk = FGSM(attack_model, loss_fn=atk_loss, eps=eps,**attack_params).perturb(batch_data.to(device),
                                                                                      batch_labels.to(device))
            if (attack_method == "PGD"):
                batch_atk = PGDAttack(attack_model, loss_fn=atk_loss, eps=eps,**attack_params).perturb(batch_data.to(device),
                                                                                       batch_labels.to(device))
            elif attack_method == "CW":
                batch_atk = CarliniWagnerL2Attack(attack_model,num_classes=10,loss_fn=atk_loss, **attack_params).perturb(batch_data.to(device),batch_labels.to(device))
            elif attack_method == "iFGSM":
                batch_atk = batch_data
                for _ in range(attack_iter):
                    batch_atk = FGSM(attack_model, loss_fn=atk_loss, eps=eps/attack_iter, **attack_params).perturb(batch_atk.to(device), batch_labels.to(device))
            x_atk = torch.cat((x_atk, batch_atk))
        # x_atk = torch.tensor(x_atk)
        self.data = x_atk.cpu()
        if quantize:
            self.data = (self.data * 255).type(torch.int) / 255.
        self.shuffle = True if data_type=='train' else False
        if shuffle is not None:
            self.shuffle = shuffle

        if for_trainning:
            self.loader = torch.utils.data.DataLoader(train_dataSet(self.noarmal_data, self.labels, self.data),
                                                      batch_size=loader_batch,shuffle=self.shuffle)
        else:
            self.loader = torch.utils.data.DataLoader(dataSet(self.data, self.labels), batch_size=loader_batch,shuffle=self.shuffle)



class cifar10_data():
    def __init__(self):
        self.num_channel = 3
        self.img_size = 32
        self.num_lables = 10
        data_train = torchvision.datasets.CIFAR10(DIR_PATH,train=True,download=True)
        data_test = torchvision.datasets.CIFAR10(DIR_PATH,train=False,download=True)
        self.train_data = torch.mul(torch.Tensor(data_train.data),1/255).permute(0,3,1,2).type(torch.FloatTensor)
        self.train_labels = torch.Tensor(data_train.targets).type(torch.long)
        self.test_data = torch.mul(torch.Tensor(data_test.data),1/255).permute(0,3,1,2).type(torch.FloatTensor)
        self.test_labels = torch.Tensor(data_test.targets).type(torch.long)

class normalCifar10():
    def __init__(self, data_type = "test",loader_batch=128,shuffle=None):
        data = cifar10_data()
        if data_type == 'train' :
            self.data = data.train_data
            self.labels = data.train_labels
            self.shuffle = True
        else :
            self.data = data.test_data
            self.labels = data.test_labels
            self.shuffle = False
        if shuffle is not None:
            self.shuffle = shuffle
        self.loader = torch.utils.data.DataLoader(dataSet(self.data, self.labels),batch_size=loader_batch,shuffle=self.shuffle)

class attackCifar10():
    def __init__(self, attack_model, attack_method="FGSM", eps=0.3, data_type="test", rand_seed=0, rand_min=0,
                 rand_max=1, loader_batch=128, for_trainning=False, atk_loss=None, quantize=False,shuffle=None,attack_params={},attack_iter=10):

        normal_data = normalCifar10(data_type=data_type, loader_batch=loader_batch)
        self.noarmal_data = normal_data.data
        self.labels = normal_data.labels

        x_atk = torch.tensor([]).to(device)
        bs = loader_batch
        for batch_data, batch_labels in normal_data.loader:
            if (attack_method == "FGSM"):
                if isinstance(eps, str):
                    batch_pn = FGSM(attack_model, loss_fn=atk_loss, getAtkpn=True,**attack_params).perturb(batch_data.to(device),
                                                                                           batch_labels.to(device))
                    eps_temp = (1) * torch.rand((len(batch_pn), 1, 1, 1))
                    eps_temp = eps_temp.to(device)
                    batch_atk = torch.clamp(batch_data.to(device) + eps_temp * batch_pn, min=0, max=1)
                else:
                    batch_atk = FGSM(attack_model, loss_fn=atk_loss, eps=eps,**attack_params).perturb(batch_data.to(device),
                                                                                      batch_labels.to(device))
            if (attack_method == "PGD"):
                batch_atk = PGDAttack(attack_model, loss_fn=atk_loss, eps=eps,**attack_params).perturb(batch_data.to(device),
                                                                                       batch_labels.to(device))
            elif attack_method == "CW":
                batch_atk = CarliniWagnerL2Attack(attack_model,num_classes=10,loss_fn=atk_loss, **attack_params).perturb(batch_data.to(device),batch_labels.to(device))
            elif attack_method == "iFGSM":
                batch_atk = batch_data
                for _ in range(attack_iter):
                    batch_atk = FGSM(attack_model, loss_fn=atk_loss, eps=eps/attack_iter, **attack_params).perturb(batch_atk.to(device), batch_labels.to(device))
            x_atk = torch.cat((x_atk, batch_atk))
        # x_atk = torch.tensor(x_atk)
        self.data = x_atk.cpu()
        if quantize:
            self.data = (self.data * 255).type(torch.int) / 255.

        self.shuffle = True if data_type=='train' else False
        if shuffle is not None:
            self.shuffle = shuffle

        if for_trainning:
            self.loader = torch.utils.data.DataLoader(train_dataSet(self.noarmal_data, self.labels, self.data),
                                                      batch_size=loader_batch,shuffle=self.shuffle)
        else:
            self.loader = torch.utils.data.DataLoader(dataSet(self.data, self.labels), batch_size=loader_batch,shuffle=self.shuffle)


class train_dataSet(torch.utils.data.Dataset):
    def __init__(self, data, labels, data_pn):
        self.data = data
        self.labels = labels
        self.data_pn = data_pn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.labels[index]
        data_pn = self.data_pn[index]
        return data, labels, data_pn


from advertorch.attacks.base import Attack, LabelMixin
from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm


# modify from advertorch.attacks.FGSM
class FGSM(Attack, LabelMixin):

    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0.,
                 clip_max=1., targeted=False, getAtkpn=False):
        """
        Create an instance of the GradientSignAttack.
        """
        super(FGSM, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        self.getAtkpn = getAtkpn

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.
        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()
        outputs = self.predict(xadv)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        grad_sign = xadv.grad.detach().sign()

        if self.getAtkpn:
            xadv = grad_sign
        else:
            xadv = xadv + self.eps * grad_sign
            xadv = clamp(xadv, self.clip_min, self.clip_max)

        return xadv.detach()