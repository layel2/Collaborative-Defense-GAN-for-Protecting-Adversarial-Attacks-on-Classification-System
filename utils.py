import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.get_data import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(model, loder_test, verbose=True, rand_seed=0):
    model.eval()
    test_loss = 0
    correct = 0
    # torch.manual_seed(rand_seed)
    with torch.no_grad():
        for data, target in loder_test:
            data = data.to(device)
            target = target.to(device)
            # print(data.shape)
            output = model(data)
            # print(output.shape,target.shape)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(loder_test.dataset)
        # test_losses.append(test_loss)
        if verbose:
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(loder_test.dataset), 100. * correct / len(loder_test.dataset)))
        return float(100. * correct / len(loder_test.dataset))


def test_gba(model, loder_test, g_ba, verbose=True, rand_seed=0):
    # torch.manual_seed(rand_seed)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loder_test:
            data = data.to(device)
            target = target.to(device)
            data = g_ba(data)
            # print(data.shape)
            output = model(data)
            # print(output.shape,target.shape)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(loder_test.dataset)
        # test_losses.append(test_loss)
        if verbose:
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(loder_test.dataset), 100. * correct / len(loder_test.dataset)))
        return float(100. * correct / len(loder_test.dataset))


def val_plot(g_ba, base_model, atk_model=None):
    pn_ratio = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    deflist, nodeflist = [], []
    base_acc = test(base_model, normalMnist().loader)
    for eps in pn_ratio:
        if atk_model is None:
            x_atk = attackMnist(base_model, eps=eps)
        else:
            x_atk = attackMnist(atk_model, eps=eps)
        temp_def = test_gba(base_model, x_atk.loader, g_ba, print_val=False)
        temp_nodef = test(base_model, x_atk.loader, print_val=False)
        deflist.append(temp_def)
        nodeflist.append(temp_nodef)
    plt.figure(figsize=(15, 7))
    plt.plot(pn_ratio, deflist)
    plt.plot(pn_ratio, nodeflist)
    plt.plot(pn_ratio, np.ones_like(pn_ratio) * base_acc)
    plt.legend(['def', 'no def', 'base acc'])
    plt.ylabel('Accuracy')
    plt.xlabel('alpha ratio')
    plt.grid(which='major')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
