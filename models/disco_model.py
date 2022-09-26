import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
    
class generator_ba(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Conv2d(in_channels=1 , out_channels=8, kernel_size=3,stride=1,padding=1)
        self.d2 = nn.Conv2d(in_channels=8 , out_channels=16, kernel_size=3,stride=1,padding=1)
        self.d3 = nn.Conv2d(in_channels=16 , out_channels=32, kernel_size=3,stride=1,padding=1)
        self.enmaxpool = nn.MaxPool2d(2)
        self.u1 = nn.Conv2d(in_channels=32,out_channels=32, kernel_size=3,padding=1)
        self.u2 = nn.Conv2d(in_channels=64,out_channels=16, kernel_size=3,padding=1)
        self.u3 = nn.Conv2d(in_channels=32,out_channels=8, kernel_size=3,padding=1)
        self.up1 = nn.Upsample(scale_factor=2)
        self.output = nn.Conv2d(in_channels=16,out_channels=1,kernel_size=3,padding=1)
        
    def forward(self,x):
        d1 = F.leaky_relu(self.d1(x), 0.2)
        x = F.max_pool2d(d1,2)
        d2 = F.instance_norm(F.leaky_relu(self.d2(x), 0.2))
        x = F.max_pool2d(d2,2)
        d3 = F.instance_norm(F.leaky_relu(self.d3(x), 0.2))
        encoder = self.enmaxpool(d3)
        x = self.up1(encoder)
        x = nn.ZeroPad2d((1,0,1,0))(x)
        x = self.u1(x)
        x = F.leaky_relu(x,0.2)
        x = F.instance_norm(x)
        u1 = torch.cat((x,d3),1)
        x = self.up1(u1)
        x = self.u2(x)
        x = F.leaky_relu(x,0.2)
        x = F.instance_norm(x)
        u2 = torch.cat((x,d2),1)
        x  = self.up1(u2)
        x = self.u3(x)
        x = F.leaky_relu(x,0.2)
        x = F.instance_norm(x)
        u3 = torch.cat((x,d1),1)
        x = self.output(u3)
        x = F.relu(x)
        return x
    
class generator_ab(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Conv2d(in_channels=1 , out_channels=8, kernel_size=3,stride=1,padding=1)
        self.d2 = nn.Conv2d(in_channels=8 , out_channels=16, kernel_size=3,stride=1,padding=1)
        self.d3 = nn.Conv2d(in_channels=16 , out_channels=32, kernel_size=3,stride=1,padding=1)
        self.enmaxpool = nn.MaxPool2d(2)
        self.u1 = nn.Conv2d(in_channels=32,out_channels=32, kernel_size=3,padding=1)
        self.u2 = nn.Conv2d(in_channels=64,out_channels=16, kernel_size=3,padding=1)
        self.u3 = nn.Conv2d(in_channels=32,out_channels=8, kernel_size=3,padding=1)
        self.up1 = nn.Upsample(scale_factor=2)
        self.output = nn.Conv2d(in_channels=16,out_channels=1,kernel_size=3,padding=1)
        
    def forward(self,x):
        d1 = F.leaky_relu(self.d1(x), 0.2)
        x = F.max_pool2d(d1,2)
        d2 = F.instance_norm(F.leaky_relu(self.d2(x), 0.2))
        x = F.max_pool2d(d2,2)
        d3 = F.instance_norm(F.leaky_relu(self.d3(x), 0.2))
        encoder = self.enmaxpool(d3)
        x = self.up1(encoder)
        x = nn.ZeroPad2d((1,0,1,0))(x)
        x = self.u1(x)
        x = F.leaky_relu(x,0.2)
        x = F.instance_norm(x)
        u1 = torch.cat((x,d3),1)
        x = self.up1(u1)
        x = self.u2(x)
        x = F.leaky_relu(x,0.2)
        x = F.instance_norm(x)
        u2 = torch.cat((x,d2),1)
        x  = self.up1(u2)
        x = self.u3(x)
        x = F.leaky_relu(x,0.2)
        x = F.instance_norm(x)
        u3 = torch.cat((x,d1),1)
        x = self.output(u3)
        x = F.tanh(x)
        return x
    
class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Conv2d(in_channels=1 , out_channels=4, kernel_size=3,stride=1,padding=1)
        self.d2 = nn.Conv2d(in_channels=4 , out_channels=8, kernel_size=3,stride=1,padding=1)
        self.d3 = nn.Conv2d(in_channels=8 , out_channels=16, kernel_size=3,stride=1,padding=1)
        self.d4 = nn.Conv2d(in_channels=16 , out_channels=32, kernel_size=3,stride=1,padding=1)
        self.val = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1,stride=1,padding=0)
        self.maxpool = nn.MaxPool2d(2)

        
    def forward(self,x):
        x = F.leaky_relu(self.d1(x),0.2)
        x = self.maxpool(x)
        x = F.instance_norm(F.leaky_relu(self.d2(x),0.2))
        x = self.maxpool(x)
        x = F.instance_norm(F.leaky_relu(self.d3(x),0.2))
        x = self.maxpool(x)
        x = F.instance_norm(F.leaky_relu(self.d4(x),0.2))
        x = self.maxpool(x)
        x = self.val(x)
        x = F.sigmoid(x)
        return x
class gen_ab_cf(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Conv2d(in_channels=3 , out_channels=8, kernel_size=3,stride=1,padding=1)
        self.d2 = nn.Conv2d(in_channels=8 , out_channels=16, kernel_size=3,stride=1,padding=1)
        self.d3 = nn.Conv2d(in_channels=16 , out_channels=32, kernel_size=3,stride=1,padding=1)
        self.enmaxpool = nn.MaxPool2d(2)
        self.u1 = nn.Conv2d(in_channels=32,out_channels=32, kernel_size=3,padding=1)
        self.u2 = nn.Conv2d(in_channels=64,out_channels=16, kernel_size=3,padding=1)
        self.u3 = nn.Conv2d(in_channels=32,out_channels=8, kernel_size=3,padding=1)
        self.up1 = nn.Upsample(scale_factor=2)
        self.output = nn.Conv2d(in_channels=16,out_channels=3,kernel_size=3,padding=1)
        
    def forward(self,x):
        d1 = F.leaky_relu(self.d1(x), 0.2)
        x = F.max_pool2d(d1,2)
        d2 = F.instance_norm(F.leaky_relu(self.d2(x), 0.2))
        x = F.max_pool2d(d2,2)
        d3 = F.instance_norm(F.leaky_relu(self.d3(x), 0.2))
        encoder = self.enmaxpool(d3)
        x = self.up1(encoder)
        x = self.u1(x)
        x = F.leaky_relu(x,0.2)
        x = F.instance_norm(x)
        u1 = torch.cat((x,d3),1)
        x = self.up1(u1)
        x = self.u2(x)
        x = F.leaky_relu(x,0.2)
        x = F.instance_norm(x)
        u2 = torch.cat((x,d2),1)
        x  = self.up1(u2)
        x = self.u3(x)
        x = F.leaky_relu(x,0.2)
        x = F.instance_norm(x)
        u3 = torch.cat((x,d1),1)
        x = self.output(u3)
        x = F.tanh(x)
        return x

class gen_ba_cf(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Conv2d(in_channels=3 , out_channels=8, kernel_size=3,stride=1,padding=1)
        self.d2 = nn.Conv2d(in_channels=8 , out_channels=16, kernel_size=3,stride=1,padding=1)
        self.d3 = nn.Conv2d(in_channels=16 , out_channels=32, kernel_size=3,stride=1,padding=1)
        self.enmaxpool = nn.MaxPool2d(2)
        self.u1 = nn.Conv2d(in_channels=32,out_channels=32, kernel_size=3,padding=1)
        self.u2 = nn.Conv2d(in_channels=64,out_channels=16, kernel_size=3,padding=1)
        self.u3 = nn.Conv2d(in_channels=32,out_channels=8, kernel_size=3,padding=1)
        self.up1 = nn.Upsample(scale_factor=2)
        self.output = nn.Conv2d(in_channels=16,out_channels=3,kernel_size=3,padding=1)
        
    def forward(self,x):
        d1 = F.leaky_relu(self.d1(x), 0.2)
        x = F.max_pool2d(d1,2)
        d2 = F.instance_norm(F.leaky_relu(self.d2(x), 0.2))
        x = F.max_pool2d(d2,2)
        d3 = F.instance_norm(F.leaky_relu(self.d3(x), 0.2))
        encoder = self.enmaxpool(d3)
        x = self.up1(encoder)
        x = self.u1(x)
        x = F.leaky_relu(x,0.2)
        x = F.instance_norm(x)
        u1 = torch.cat((x,d3),1)
        x = self.up1(u1)
        x = self.u2(x)
        x = F.leaky_relu(x,0.2)
        x = F.instance_norm(x)
        u2 = torch.cat((x,d2),1)
        x  = self.up1(u2)
        x = self.u3(x)
        x = F.leaky_relu(x,0.2)
        x = F.instance_norm(x)
        u3 = torch.cat((x,d1),1)
        x = self.output(u3)
        x = F.relu(x)
        return x

class dis_cf(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Conv2d(in_channels=3 , out_channels=8, kernel_size=3,stride=1,padding=1)
        self.d2 = nn.Conv2d(in_channels=8 , out_channels=16, kernel_size=3,stride=1,padding=1)
        self.d3 = nn.Conv2d(in_channels=16 , out_channels=32, kernel_size=3,stride=1,padding=1)
        self.d4 = nn.Conv2d(in_channels=32 , out_channels=64, kernel_size=3,stride=1,padding=1)
        self.d5 = nn.Conv2d(in_channels=64 , out_channels=128, kernel_size=3,stride=1,padding=1)
        self.val = nn.Conv2d(in_channels=128,out_channels=1,kernel_size=1,stride=1,padding=0)
        self.maxpool = nn.MaxPool2d(2)

        
    def forward(self,x):
        x = F.leaky_relu(self.d1(x),0.2)
        x = self.maxpool(x)
        x = F.instance_norm(F.leaky_relu(self.d2(x),0.2))
        x = self.maxpool(x)
        x = F.instance_norm(F.leaky_relu(self.d3(x),0.2))
        x = self.maxpool(x)
        x = F.instance_norm(F.leaky_relu(self.d4(x),0.2))
        x = self.maxpool(x)
        x = F.instance_norm(F.leaky_relu(self.d5(x),0.2))
        x = self.maxpool(x)
        x = self.val(x)
        x = F.sigmoid(x)
        return x