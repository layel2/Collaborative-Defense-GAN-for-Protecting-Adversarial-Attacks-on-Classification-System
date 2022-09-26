import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import itertools
from tqdm import tqdm

from data.get_data import *
from utils import *
from models.cifar10_model import *
from models.mnist_model import *
from models.disco_model import *

import argparse

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str ,help='Dataset ("mnist", "fmnist" and "cifar10") ',required=True)
parser.add_argument('--arch',type=str ,help="Classifier architecture A/B/C for mnist/fmnist A/B for cifar10",default='A')
parser.add_argument('--clf_model',type=str, help="Classifier model path", default=None)
parser.add_argument('--n_epochs',type=int,help="Number of training epochs (default 10)" ,default=10)
parser.add_argument('--batch_size',type=int, help="Number of batch size" ,default=128)
parser.add_argument('--lr',type=float,help="Number of learning rate (default 0.0002) " ,default=0.0002)
parser.add_argument('-o', '--output_dirs',type=str, help="Output save directory", default=None)

args = parser.parse_args()

if args.output_dirs is None:
    output_dir = f"./saved_model/collaborative_gan_{args.dataset}_{args.arch}/"
else:
    output_dir = args.output_dirs

if args.clf_model is None:
    clf_model_path = f"./saved_model/{args.dataset}_{args.arch}.pth"
else:
    clf_model_path = args.clf_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Working on {device}")

def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad

def train(loader_train,base_model,num_epoch):
    start_time = datetime.datetime.now()
    for epoch in tqdm(range(num_epoch)):
        for i,(a_real,label) in enumerate(loader_train):
            #Generator
            a_real = a_real.to(device)
            label = label.to(device)
            alpha = (1)*torch.rand(1)
            alpha = alpha.to(device)
            
            a_real = torch.autograd.Variable(a_real)
            b_real = FGSM(base_model,loss_fn=torch.nn.NLLLoss(),eps=alpha).perturb(a_real,label)
            
            set_grad([d_a,d_b,base_model], False)
            g_opt.zero_grad()
            
            fake_b_pn = g_ab(a_real)
            fake_b = alpha*fake_b_pn + a_real
            fake_b = torch.clamp(fake_b,min=0,max=1)
            fake_a = g_ba(b_real)
            
            recon_a = g_ba(fake_b)
            recon_b_pn = g_ab(fake_a)
            recon_b = torch.clamp(alpha*recon_b_pn + fake_a,min=0,max=1)
            
            a_fake_dis = d_a(fake_a)
            b_fake_dis = d_b(fake_b)
            
            a_id = g_ba(a_real)
            
            #Gen loss
            real_label = torch.autograd.Variable(torch.ones(a_fake_dis.size()).to(device))
            a_dis_loss = mae_loss(a_fake_dis,real_label)
            b_dis_loss = mae_loss(b_fake_dis,real_label)
            
            a_fake_loss = mse_loss(fake_a,a_real)
            b_fake_loss = mse_loss(fake_b,b_real)*2
            
            a_recon_loss = mse_loss(recon_a,a_real)
            b_recon_loss = mse_loss(recon_b,b_real)*2
            
            
            a_id_loss = mse_loss(a_id,a_real)
            a_pred_loss = cel_loss(base_model(fake_a),label)
            
            #total gen loss
            g_loss = a_dis_loss+b_dis_loss+a_fake_loss+b_fake_loss+a_recon_loss+b_recon_loss+a_pred_loss+a_id_loss
            g_loss.backward()
            g_opt.step()
            #Discriminator
            set_grad([d_a,d_b], True)
            d_opt.zero_grad()

            fake_b_pn = g_ab(a_real)
            fake_b = alpha*fake_b_pn + a_real
            fake_b = torch.clamp(fake_b,min=0,max=1)
            fake_a = g_ba(b_real)
            
            a_real_dis = d_a(a_real)
            b_real_dis = d_b(b_real)
            a_fake_dis = d_a(fake_a)
            b_fake_dis = d_b(fake_b)
            
            real_label = torch.autograd.Variable(torch.ones(a_fake_dis.size()).to(device))
            fake_label = torch.autograd.Variable(torch.zeros(a_fake_dis.size()).cuda())
            
            a_dis_real_loss = mae_loss(a_real_dis,real_label)
            a_dis_fake_loss = mae_loss(a_fake_dis,fake_label)
            a_dis_loss = (a_dis_real_loss+a_dis_fake_loss)*0.5
            
            b_dis_real_loss = mae_loss(b_real_dis,real_label)
            b_dis_fake_loss = mae_loss(b_fake_dis,fake_label)
            b_dis_loss = (b_dis_real_loss + b_dis_fake_loss)*0.5
            
            
            a_dis_loss.backward()
            b_dis_loss.backward()
            d_opt.step()
            
            elapsed_time = datetime.datetime.now() - start_time
            
        print ("[%d] [%d/%d] time: %s, [d_loss: %f, g_loss: %f]" % (epoch, i+1,
                                                                      len(loader_train),
                                                                      elapsed_time,
                                                                      a_dis_loss+b_dis_loss, g_loss))
        my_log.append([a_fake_loss,b_fake_loss,a_recon_loss,b_recon_loss,a_pred_loss,a_id_loss])
        g_loss_log.append(g_loss)
        d_loss_log.append([a_dis_loss+b_dis_loss,a_dis_loss,b_dis_loss])
        dcom_log.append([a_dis_real_loss,a_dis_fake_loss,b_dis_real_loss,b_dis_fake_loss])
            
        test_log.append(test_gba(base_model,data_test_atk.loader,g_ba, verbose=False))
    
clf_model_mapper = {
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

data_atk_mapper = {
    "mnist": attackMnist,
    "fmnist": attackFMnist,
    "cifar10": attackCifar10,
}

if __name__ == '__main__':
    clf_model = clf_model_mapper[f"{args.dataset}_{args.arch}"]().to(device)
    clf_model.load_state_dict(torch.load(clf_model_path))
    if args.dataset == 'cifar10':
        g_ab = gen_ab_cf().to(device)
        g_ba = gen_ba_cf().to(device)
        d_a = dis_cf().to(device)
        d_b = dis_cf().to(device)
    else:
        g_ab = generator_ab().to(device)
        g_ba = generator_ba().to(device)
        d_a = discriminator().to(device)
        d_b = discriminator().to(device)
    
    data_train = data_mapper[f"{args.dataset}"](data_type='train', loader_batch=args.batch_size)
    loader_train = data_train.loader
    data_test_atk = data_atk_mapper[f"{args.dataset}"](clf_model,attack_method="FGSM",data_type = "test",atk_loss=nn.NLLLoss(), loader_batch=args.batch_size)

    g_opt = optim.Adam(itertools.chain(g_ab.parameters(),g_ba.parameters()) ,lr=args.lr, betas=(0.5, 0.999))
    d_opt = optim.Adam(itertools.chain(d_a.parameters(),d_b.parameters()) ,lr=args.lr, betas=(0.5, 0.999))
    mae_loss = nn.L1Loss().to(device)
    mse_loss = nn.MSELoss().to(device)
    cel_loss = nn.CrossEntropyLoss().to(device)

    d_loss_log = []
    g_loss_log = []
    my_log = []
    dcom_log = []
    test_log = []
    train(loader_train, clf_model, args.n_epochs)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    torch.save(g_ab.state_dict(), os.path.join(output_dir,'g_ab.pth'))
    torch.save(g_ba.state_dict(), os.path.join(output_dir,'g_ba.pth'))
    torch.save(d_a.state_dict(), os.path.join(output_dir,'d_a.pth'))
    torch.save(d_b.state_dict(), os.path.join(output_dir,'d_b.pth'))
