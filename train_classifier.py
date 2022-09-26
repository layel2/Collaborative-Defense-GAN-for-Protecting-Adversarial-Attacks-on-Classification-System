import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from data.get_data import *
from utils import *
from models.mnist_model import *
from models.cifar10_model import *

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str , help='Dataset ("mnist", "fmnist" and "cifar10") ',required=True)
parser.add_argument('--arch',type=str , help="Classifier architecture A/B/C for mnist/fmnist A/B for cifar10",default='A')
parser.add_argument('--n_epochs',type=int, help="Number of training epochs" ,default=10)
parser.add_argument('--batch_size',type=int, help="Number of batch size" ,default=128)
parser.add_argument('-o', '--output',type=str, help="Output save path", default=None)

args = parser.parse_args()

if args.output is None:
    output_path = f"./saved_model/{args.dataset}_{args.arch}.pth"
else:
    output_path = args.output

def train(epoch):
    model.train()
    for batch_idx, (batch_data, batch_target) in enumerate(loder_train):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)
        optimizer.zero_grad()
        output = model(batch_data)
        loss = cel_loss(output, batch_target)
        loss.backward()
        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(batch_data), len(loder_train.dataset),
        #         100. * batch_idx / len(loder_train), loss.item()))
        train_losses.append(loss.item())
        train_counter.append((batch_idx*64) + ((epoch-1)*len(loder_train.dataset)))

def test(model,loder_test):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loder_test:
            data = data.to(device)
            target = target.to(device)
            #print(data.shape)
            output = model(data)
            #print(output.shape,target.shape)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(loder_test.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(loder_test.dataset),100. * correct / len(loder_test.dataset)))

model_mapper = {
    "mnist_A": mnistmodel_A,
    "mnist_B": mnistmodel_B,
    "mnist_C": mnistmodel_C,
    "fmnist_A": mnistmodel_A,
    "fmnist_B": mnistmodel_B,
    "fmnist_C": mnistmodel_C,
    "cifar_A": cifar10_a,
    "cifar_B": cifar10_b,
}

data_mapper = {
    "mnist": normalMnist,
    "fmnist": normalFMnist,
    "cifar10": normalCifar10,
}

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"working on {device}")
    model = model_mapper[f"{args.dataset}_{args.arch}"]().to(device)
    data = data_mapper[f"{args.dataset}"](data_type='train', loader_batch=args.batch_size)
    data_test = data_mapper[f"{args.dataset}"](data_type='test', loader_batch=args.batch_size)
    train_losses = []
    train_counter = []
    test_losses = []
    n_epochs = args.n_epochs
    test_counter = [i*len(data.loader.dataset) for i in range(n_epochs + 1)]
    optimizer = torch.optim.Adam(model.parameters(),eps=10e-7,lr=0.0001)
    log_interval = 10
    cel_loss = nn.CrossEntropyLoss()
    loder_train = data.loader
    for epoch in tqdm(range(1, n_epochs + 1)):
        train(epoch)
        test(model,data_test.loader)
    if not os.path.isdir(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torch.save(model.state_dict(), output_path)
    print("done")