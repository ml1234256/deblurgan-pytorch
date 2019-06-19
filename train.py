import torch
from torch.autograd import Variable
# import matplotlib.pyplot as plt
from utils import ReadConcat, image_transform, image_recovery, update_lr, check_folder, save_image, save_net
from losses import get_loss
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def train(opt, netG, netD, optim_G, optim_D):
    tensor = torch.cuda.FloatTensor
    # lossD_list = []
    # lossG_list = []

    train = ReadConcat(opt)
    trainset = DataLoader(train, batch_size=opt.batchSize, shuffle=True)
    save_img_path = os.path.join('./result', 'train')
    check_folder(save_img_path)

    for e in range(opt.epoch, opt.niter + opt.niter_decay + 1):
        for i, data in enumerate(trainset):
            # set input
            data_A = data['A'] # blur
            data_B = data['B'] #sharp
            # plt.imshow(image_recovery(data_A.squeeze().numpy()))
            # plt.pause(0)
            # print(data_A.shape)
            # print(data_B.shape)

            if torch.cuda.is_available():
                data_A = data_A.cuda(opt.gpu)
                data_B = data_B.cuda(opt.gpu)
            # forward
            realA = Variable(data_A)
            fakeB = netG(realA)
            realB = Variable(data_B)

            # optimize_parameters
            # optimizer netD
            set_requires_grad([netD], True)
            for iter_d in range(1):
                optim_D.zero_grad()
                loss_D, _ = get_loss(tensor, netD, realA, fakeB, realB)
                loss_D.backward()
                optim_D.step()

            # optimizer netG
            set_requires_grad([netD], False)
            optim_G.zero_grad()
            _, loss_G = get_loss(tensor, netD, realA, fakeB, realB)
            loss_G.backward()
            optim_G.step()
            if i % 50 == 0:
                # lossD_list.append(loss_D)
                # lossG_list.append(loss_G)
                print('{}/{}: lossD:{}, lossG:{}'.format(i, e, loss_D, loss_G))

        visul_img = torch.cat((realA, fakeB, realA), 3)
        #print(type(visul_img), visul_img.size())
        visul_img = image_recovery(visul_img)
        #print(visul_img.size)
        save_image(visul_img, os.path.join(save_img_path,'epoch'+str(e)+'.png'))

        if e > opt.niter:
            update_lr(optim_D, opt.lr, opt.niter_decay)
            lr = (optim_G, opt.lr, opt.niter_decay)
            opt.lr = lr

        if e % opt.save_epoch_freq == 0:
            save_net(netG, opt.checkpoints_dir, 'G', e)
            save_net(netD, opt.checkpoints_dir, 'D', e)
