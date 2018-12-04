# coding:utf-8
import argparse
import os
import numpy as np
import math
import time

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch

import pandas as pd
import matplotlib.pyplot as plt

from models import *
from scipy.stats import norm


os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--lr2', type=float, default=0.0002, help='adam: learning rate2')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
# parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
parser.add_argument('--dataset', type=str, default='mnist', help='Select dataset')
parser.add_argument('--resuem', '-r', action = 'store_true', help='resume from checkpoint')
opt = parser.parse_args()
print(opt)

if opt.dataset == 'mnist':
    channels = 1
else:
    channels = 3

img_shape = (channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Real or fake
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**4
        self.adv_layer = nn.Sequential( nn.Linear(128*ds_size**2, 1),
                                        nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
#####################################

####### オリジナル損失関数 ########
class ConfusedLoss(nn.Module):
    def __init__(self):
        super(ConfusedLoss, self).__init__()

    def forward(self, p, mu, sigma):
        p_data = torch.zeros(len(p))

        # 2乗誤差を利用したLoss
        # for i in range (len(p)):
            # p_data[i] = (torch.tensor(0.5 - torch.topk(p, k=2)[0][i][1] ))**2 *10

        # 正規分布を利用したLoss
        pi = math.pi
        e = math.e
        sigma2 = sigma**2
        for i in range (len(p)):
            x = torch.topk(p, k=2)[0][i][1] # 2番めに高い確率
            norm_pdf =  (1 / math.sqrt(2*pi*sigma2))*e**(-((mu-mu)**2/(2*sigma2)))-(1 / math.sqrt(2*pi*sigma2))*e**(-((x-mu)**2/(2*sigma2)))
            p_data[i] = torch.tensor(norm_pdf) * 10
        loss = torch.sum(p_data) / len(p)

        return loss
#####################################




# Loss function:Binary cross enrtopy criterion
adversarial_loss = torch.nn.BCELoss()
confuse_loss = ConfusedLoss()


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
discriminator2 = VGG('VGG19', channels=channels, num_class=10)

print(generator)
print(discriminator)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
discriminator2 = discriminator2.to(device)
if device == 'cuda':
    discriminator2= torch.nn.DataParallel(discriminator2)
    cudnn.benchmark = True

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    confuse_loss.cuda()
    print("CUDA")

# Load VGG ckpt.7
discriminator2Path = './checkpoint/' + opt.dataset + '/VGG19/'
print('==> Resuming from discriminator2 checkpoint...')
assert os.path.isdir(discriminator2Path), 'Error:no Generator`s checkpoint directory found!, You did not train with ' + opt.dataset + '!!'
discriminator2.load_state_dict(torch.load(discriminator2Path + 'ckpt.t7'))
print('------ OK ------')

generatorPath = './checkpoint/' + opt.dataset + '/generator/'
discriminatorPath = './checkpoint/' + opt.dataset + '/discriminator/'

if opt.resuem:
    print('==> Resuming from generator and discriminator checkpoint...')
    assert os.path.isdir(generatorPath), 'Error:no Generator`s checkpoint directory found!, You did not train with ' + opt.dataset + '!!'
    assert os.path.isdir(discriminatorPath), 'Error:no Discriminator`s checkpoint directory found!, You did not train with ' + opt.dataset + '!!'
    generator.load_state_dict(torch.load(generatorPath + '/generator.t7'))
    discriminator.load_state_dict(torch.load(discriminatorPath + '/discriminator.t7'))
    print('------ OK ------')

print('==> Preparing data..')

# MNIST
if (opt.dataset == 'mnist'):
    os.makedirs('./data/mnist', exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(opt.img_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=opt.batch_size, shuffle=True)

# cifar10
elif opt.dataset == 'cifar10':
    os.makedirs('./data/cifar10', exist_ok=True)
    transform_data = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True) # , num_workers=2)

else:
    print("See you")
    exit()
print('------ OK ------')

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G2 = torch.optim.Adam(generator.parameters(), lr=opt.lr2, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Logging
logs = []

# epoch用　csv化のみ
logs2 = []

# ----------
#  Training
# ----------
print('==> Start Training...')
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        # print(len(real_imgs)) # 64

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        # print(discriminator(gen_imgs)) # 64個の real or fake結果
        # print(g_loss) # tensor(0.790)
        g_loss.backward(retain_graph=True)
        # g_loss.backward()
        optimizer_G.step()

        #Discriminator2に,2番目に大きいデータを50%に近づけ, 再びGeneratorのパラメータを更新する
        optimizer_G2.zero_grad()
        discriminator2.eval() #testモード
        p = F.softmax(discriminator2(gen_imgs))
        # print(np.sort(p.data[1])[-2]) # 2番めに大きい値
        # print(p.data[1][1])
        # print(np.argsort(p.data[1], axis=1))

        c_loss = confuse_loss(p, mu=0.5, sigma=0.1)
        c_loss.backward(retain_graph=True)
        c_loss.backward()
        # gc_loss = (g_loss + c_loss) / 2
        # gc_loss.backward()
        # opttimizer_G.step()
        optimizer_G2.step()
        # exit()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        # fake imageを通じて勾配がＧに伝わらないようにdetach()して止める
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [C loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), c_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            imagePath1 = 'images/'+ opt.dataset +'/iteration/'
            os.makedirs(imagePath1, exist_ok=True)
            print('---Save Image---')
            save_image(gen_imgs.data[:64], imagePath1 + '%d.png' % batches_done, nrow=8, normalize=True)

        # Logging for iteration
        log = {'iteration':batches_done, 'D_Loss':d_loss.item(), 'G_Loss':g_loss.item(), 'C_Loss':c_loss.item()}
        logs.append(log)
        ############## Batch 処理 ##############

    #Logging for epoch
    log2 = {'epoch':epoch, 'D_Loss':d_loss.item(), 'G_Loss':g_loss.item(), 'C_Loss':c_loss.item()}
    logs2.append(log2)

    # save Image
    imagePath2 = 'images/'+ opt.dataset +'/epoch/'
    os.makedirs(imagePath2, exist_ok=True)
    print('---Save Image---')
    save_image(gen_imgs.data[:64], imagePath2 + '%d.png' % epoch, nrow=8, normalize=True)
############## epoch ##############

# 1枚ずつ保存
imagePath3 = 'images/'+ opt.dataset +'/final_images/'
os.makedirs(imagePath3, exist_ok=True)
for i in range (len(gen_imgs)):
# print(len(gen_imgs.data[:64])) # 先頭から64まで取得
    save_image(gen_imgs.data[i],  imagePath3 + 'epoch%d_%d.png' % (epoch, i), normalize=True)

#save csv
graphPath = './graph/' + opt.dataset + '/'
os.makedirs(graphPath, exist_ok=True)

# iteration
df_logs = pd.DataFrame(logs)
df_logs.to_csv(graphPath + 'log_iteration.csv', index = False)
# epoch
df_logs2 = pd.DataFrame(logs2)
df_logs2.to_csv(graphPath + 'log_epoch.csv', index = False)


#save graph
x = df_logs['iteration'].values
y0 = df_logs['D_Loss'].values
y1 = df_logs['G_Loss'].values
y2 = df_logs['C_Loss'].values

fig, ax = plt.subplots(1,3, figsize=(24,8)) # 1x3 のグラフを生成
ax[0].plot(x, y0, label='D_Loss', color = "red")
ax[0].legend()
ax[1].plot(x, y1, label='G_Loss', color = "blue")
ax[1].legend()
ax[2].plot(x, y2, label='C_Loss', color = "green")
ax[2].legend()

plt.savefig(graphPath + 'graph.png')
# plt.show()


# save paramaer
generatorPath = './checkpoint/' + opt.dataset + '/generator/'
discriminatorPath = './checkpoint/' + opt.dataset + '/discriminator/'
os.makedirs(generatorPath, exist_ok=True)
os.makedirs(discriminatorPath, exist_ok = True)
torch.save(generator.state_dict(), generatorPath + '/generator.t7')
torch.save(discriminator.state_dict(), discriminatorPath + '/discriminator.t7')

finish_time_min = time.time() - start // 60
finish_time_sec = time.time() - start % 60

print('Finish Time: {0}[min] {1}[sec]'.format(finish_time_min, finish_time_sec))
